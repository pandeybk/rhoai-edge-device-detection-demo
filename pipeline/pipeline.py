# pipeline.py
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath
from kfp import kubernetes
from typing import NamedTuple

from modules.data_preparation import data_preparation
from modules.model_initialization import model_initialization
from modules.model_training import model_training
from modules.model_evaluation import model_evaluation
from modules.upload_model import upload_model
from modules.run_openvino_model_server import run_openvino_model_server
from modules.run_inference import run_inference

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def data_preparation_component(s3_dataset_path: str, 
                               batch_size: int, 
                               train_loader_path: OutputPath(), 
                               test_loader_path: OutputPath()) -> NamedTuple('Outputs', [('output_size', int)]):
    import os
    import json
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from PIL import Image
    from collections import Counter
    import boto3
    import botocore
    import logging
    import subprocess
    from typing import NamedTuple

    class CocoFormatDataset(torch.utils.data.Dataset):
        def __init__(self, json_file, root_dir, transform=None, required_annotations=8):
            with open(json_file, 'r') as f:
                self.coco_data = json.load(f)

            image_id_to_anns = Counter(ann['image_id'] for ann in self.coco_data['annotations'])
            valid_image_ids = {id for id, count in image_id_to_anns.items() if count == required_annotations}

            self.images = [img for img in self.coco_data['images'] if img['id'] in valid_image_ids]
            self.annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] in valid_image_ids]
            self.root_dir = root_dir
            self.transform = transform
            self.cat_id_to_label = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}

            logging.debug(f"All image ids: {image_id_to_anns}")
            logging.debug(f"Valid image ids: {valid_image_ids}")
            logging.debug(f"Dataset initialized with {len(self.images)} images and {len(self.annotations)} annotations")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_info = self.images[idx]
            img_path = os.path.join(self.root_dir, 'images', img_info['file_name'])
            image = Image.open(img_path).convert('RGB')

            ann_ids = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
            labels = torch.zeros(len(self.cat_id_to_label), dtype=torch.float32)
            categories = []

            for ann in ann_ids:
                cat_id = ann['category_id']
                label_index = self.cat_id_to_label[cat_id]
                labels[label_index] = 1
                for cat in self.coco_data['categories']:
                    if cat['id'] == cat_id:
                        category_name = cat['name']
                        categories.append(category_name)
                        break

            if self.transform:
                image = self.transform(image)

            if len(categories) != 8:
                logging.warning(f"Warning: Expected 8 categories for image_id {img_info['id']}, got {len(categories)}. Skipping.")
                return None

            return image, labels

    def data_preparation(s3_dataset_path: str, 
                        batch_size: int, 
                        train_loader_path: str, 
                        test_loader_path: str) -> NamedTuple('Outputs', [('output_size', int)]):

        # Enable logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Install required packages
        packages = ["boto3", "botocore", "torch", "torchvision", "pillow"]
        for package in packages:
            logger.debug(f"Installing package: {package}")
            subprocess.run(["pip", "install", package], check=True)

        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
        region_name = os.environ.get('AWS_DEFAULT_REGION')
        bucket_name = os.environ.get('AWS_S3_BUCKET')

        logger.debug(f"Connecting to S3 with endpoint: {endpoint_url}, region: {region_name}, bucket: {bucket_name}")

        session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                        aws_secret_access_key=aws_secret_access_key)

        s3_resource = session.resource(
            's3',
            config=botocore.client.Config(signature_version='s3v4'),
            endpoint_url=endpoint_url,
            region_name=region_name)

        bucket = s3_resource.Bucket(bucket_name)
        
        # Create directories
        root_dir = '/tmp/data'
        images_dir = os.path.join(root_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Download images from S3
        logger.debug(f"Downloading images from {s3_dataset_path}/images/")
        for obj in bucket.objects.filter(Prefix=f"{s3_dataset_path}/images/"):
            if obj.key.endswith("/"):
                continue
            file_name = obj.key.split('/')[-1]
            logger.debug(f"Downloading {obj.key} to {os.path.join(images_dir, file_name)}")
            bucket.download_file(obj.key, os.path.join(images_dir, file_name))

        # Download annotations from S3
        annotations_key = f"{s3_dataset_path}/result.json"
        logger.debug(f"Downloading annotations from {annotations_key}")
        try:
            bucket.download_file(annotations_key, os.path.join(root_dir, 'result.json'))
        except botocore.exceptions.ClientError as e:
            logger.error(f"Error downloading annotations: {e}")
            raise

        # Load annotations
        json_file = os.path.join(root_dir, 'result.json')
        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        # Transformations
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def prepare_dataset(json_file, root_dir, train_transform, test_transform):
            full_dataset = CocoFormatDataset(json_file=json_file, root_dir=root_dir, transform=train_transform)
            logger.debug(f"Full dataset size: {len(full_dataset)}")
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
            test_dataset.dataset.transform = test_transform
            return train_dataset, test_dataset

        def get_data_loaders(train_dataset, test_dataset, batch_size):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            logger.debug(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")
            return train_loader, test_loader

        train_dataset, test_dataset = prepare_dataset(json_file, root_dir, train_transform, test_transform)
        train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

        torch.save(train_loader, train_loader_path)
        torch.save(test_loader, test_loader_path)

        output_size = len(train_dataset.dataset.cat_id_to_label)
        return (output_size,)

    return data_preparation(s3_dataset_path, batch_size, train_loader_path, test_loader_path)

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def model_initialization_component(output_size: int, model_output_path: OutputPath()):
    model_initialization(output_size, model_output_path)

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def model_training_component(train_loader_path: InputPath(), num_epochs: int, learning_rate: float, weight_decay: float, model_output_path: OutputPath()):
    model_training(train_loader_path, num_epochs, learning_rate, weight_decay, model_output_path)

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def model_evaluation_component(model_path: InputPath(), test_loader_path: InputPath()):
    model_evaluation(model_path, test_loader_path)

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def upload_model_component(model_path: InputPath(), s3_key: str):
    upload_model(model_path, s3_key)

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def run_openvino_model_server_component(model_path: InputPath(), server_url: OutputPath(), token: OutputPath()):
    run_openvino_model_server(model_path, server_url, token)

@dsl.component(base_image="quay.io/bpandey/kfp-pipeline:v0.0.1")
def run_inference_component(server_url: str, token: str, inference_results: OutputPath()):
    run_inference(server_url, token, inference_results)

@dsl.pipeline(name='Image Detection Pipeline')
def image_detection_pipeline(s3_dataset_path: str = 'dataset/custom-dataset',
                             batch_size: int = 4,
                             num_epochs: int = 20,
                             learning_rate: float = 0.001,
                             weight_decay: float = 1e-4,
                             s3_key: str = 'models/router-detection/1/model.onnx'):

    # Data preparation step
    data_preparation_task = data_preparation_component(s3_dataset_path=s3_dataset_path, 
                                                       batch_size=batch_size)

    kubernetes.use_secret_as_env(
        task=data_preparation_task,
        secret_name='s3-connection-edge-router-detection',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
    })

    # Model initialization step
    model_initialization_task = model_initialization_component(output_size=data_preparation_task.outputs['output_size'])

    # Model training step
    # model_training_task = model_training_component(train_loader_path=data_preparation_task.outputs['train_loader_path'], 
    #                                                num_epochs=num_epochs, 
    #                                                learning_rate=learning_rate, 
    #                                                weight_decay=weight_decay, 
    #                                                model_output_path=model_initialization_task.outputs['model_output_path'])

    # # Model evaluation step
    # model_evaluation_task = model_evaluation_component(model_path=model_training_task.outputs['model_output_path'], 
    #                                                    test_loader_path=data_preparation_task.outputs['test_loader_path'])

    # # Model upload step
    # upload_model_task = upload_model_component(model_path=model_training_task.outputs['model_output_path'], s3_key=s3_key)
    # upload_model_task.set_env_variable(name="S3_KEY", value=s3_key)

    # kubernetes.use_secret_as_env(
    #     task=upload_model_task,
    #     secret_name='s3-connection-edge-router-detection',
    #     secret_key_to_env={
    #         'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
    #         'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
    #         'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
    #         'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
    #         'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
    #     })

    # # Run OpenVINO model server
    # run_openvino_model_server_task = run_openvino_model_server_component(model_path=upload_model_task.outputs['model_output_path'])

    # # Run inference
    # run_inference_task = run_inference_component(server_url=run_openvino_model_server_task.outputs['server_url'], 
    #                                              token=run_openvino_model_server_task.outputs['token'], 
    #                                              inference_results="inference_results_path")

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=image_detection_pipeline,
        package_path='image_detection_pipeline.yaml'
    )
