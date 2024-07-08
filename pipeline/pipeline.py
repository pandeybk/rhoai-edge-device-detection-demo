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

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
def data_preparation_component(s3_dataset_path: str, 
                               batch_size: int, 
                               train_loader_path: OutputPath(), 
                               test_loader_path: OutputPath()) -> NamedTuple('Outputs', [('output_size', int)]):
    return data_preparation(s3_dataset_path, batch_size, train_loader_path, test_loader_path)

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
def model_initialization_component(output_size: int, model_output_path: OutputPath()):
    model_initialization(output_size, model_output_path)

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
def model_training_component(train_loader_path: InputPath(), num_epochs: int, learning_rate: float, weight_decay: float, model_output_path: OutputPath()):
    model_training(train_loader_path, num_epochs, learning_rate, weight_decay, model_output_path)

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
def model_evaluation_component(model_path: InputPath(), test_loader_path: InputPath()):
    model_evaluation(model_path, test_loader_path)

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
def upload_model_component(model_path: InputPath(), s3_key: str):
    upload_model(model_path, s3_key)

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
def run_openvino_model_server_component(model_path: InputPath(), server_url: OutputPath(), token: OutputPath()):
    run_openvino_model_server(model_path, server_url, token)

@dsl.component(base_image="python:3.11", target_image='bkpandey/kfp-pipeline', packages_to_install=["boto3", "botocore", "torch", "torchvision", "pillow"])
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
