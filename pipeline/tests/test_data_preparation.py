import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from modules.data_preparation import data_preparation

class TestDataPreparation(unittest.TestCase):

    @patch('modules.data_preparation.boto3')
    @patch('builtins.open', new_callable=mock_open)
    def test_data_preparation(self, mock_open, mock_boto3):
        # Setup
        mock_s3 = MagicMock()
        mock_boto3.session.Session.return_value.resource.return_value.Bucket.return_value = mock_s3

        mock_s3.objects.filter.return_value = [
            MagicMock(key='dataset/custom-dataset/images/028d2ad5-IMG_4850.jpg'),
            MagicMock(key='dataset/custom-dataset/images/04180cf3-IMG_4875.jpg')
        ]

        # Mock result.json content
        result_json = {
            "images": [
                {"width": 3024, "height": 4032, "id": 0, "file_name": "028d2ad5-IMG_4850.jpg"},
                {"width": 3024, "height": 4032, "id": 1, "file_name": "04180cf3-IMG_4875.jpg"}
            ],
            "categories": [
                {"id": 0, "name": "ConsolePort_Connected"},
                {"id": 1, "name": "ConsolePort_NotConnected"},
                {"id": 2, "name": "Category_3"},
                {"id": 3, "name": "Category_4"},
                {"id": 4, "name": "Category_5"},
                {"id": 5, "name": "Category_6"},
                {"id": 6, "name": "Category_7"},
                {"id": 7, "name": "Category_8"}
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 0, "segmentation": [[939.909, 560.741, 1345.780, 582.103]], "bbox": [939.909, 560.741, 405.870, 304.402], "ignore": 0, "iscrowd": 0, "area": 107334.432},
                {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[1041.377, 939.909, 1361.801, 955.931]], "bbox": [1041.377, 939.909, 363.147, 347.125], "ignore": 0, "iscrowd": 0, "area": 112967.101},
                {"id": 2, "image_id": 0, "category_id": 2, "segmentation": [[1100.377, 940.909, 1365.801, 960.931]], "bbox": [1100.377, 940.909, 365.147, 349.125], "ignore": 0, "iscrowd": 0, "area": 113000.101},
                {"id": 3, "image_id": 0, "category_id": 3, "segmentation": [[1110.377, 945.909, 1370.801, 965.931]], "bbox": [1110.377, 945.909, 367.147, 350.125], "ignore": 0, "iscrowd": 0, "area": 113500.101},
                {"id": 4, "image_id": 0, "category_id": 4, "segmentation": [[1120.377, 950.909, 1375.801, 970.931]], "bbox": [1120.377, 950.909, 369.147, 352.125], "ignore": 0, "iscrowd": 0, "area": 114000.101},
                {"id": 5, "image_id": 0, "category_id": 5, "segmentation": [[1130.377, 955.909, 1380.801, 975.931]], "bbox": [1130.377, 955.909, 371.147, 354.125], "ignore": 0, "iscrowd": 0, "area": 114500.101},
                {"id": 6, "image_id": 0, "category_id": 6, "segmentation": [[1140.377, 960.909, 1385.801, 980.931]], "bbox": [1140.377, 960.909, 373.147, 356.125], "ignore": 0, "iscrowd": 0, "area": 115000.101},
                {"id": 7, "image_id": 0, "category_id": 7, "segmentation": [[1150.377, 965.909, 1390.801, 985.931]], "bbox": [1150.377, 965.909, 375.147, 358.125], "ignore": 0, "iscrowd": 0, "area": 115500.101},
                {"id": 8, "image_id": 1, "category_id": 0, "segmentation": [[939.909, 560.741, 1345.780, 582.103]], "bbox": [939.909, 560.741, 405.870, 304.402], "ignore": 0, "iscrowd": 0, "area": 107334.432},
                {"id": 9, "image_id": 1, "category_id": 1, "segmentation": [[1041.377, 939.909, 1361.801, 955.931]], "bbox": [1041.377, 939.909, 363.147, 347.125], "ignore": 0, "iscrowd": 0, "area": 112967.101},
                {"id": 10, "image_id": 1, "category_id": 2, "segmentation": [[1100.377, 940.909, 1365.801, 960.931]], "bbox": [1100.377, 940.909, 365.147, 349.125], "ignore": 0, "iscrowd": 0, "area": 113000.101},
                {"id": 11, "image_id": 1, "category_id": 3, "segmentation": [[1110.377, 945.909, 1370.801, 965.931]], "bbox": [1110.377, 945.909, 367.147, 350.125], "ignore": 0, "iscrowd": 0, "area": 113500.101},
                {"id": 12, "image_id": 1, "category_id": 4, "segmentation": [[1120.377, 950.909, 1375.801, 970.931]], "bbox": [1120.377, 950.909, 369.147, 352.125], "ignore": 0, "iscrowd": 0, "area": 114000.101},
                {"id": 13, "image_id": 1, "category_id": 5, "segmentation": [[1130.377, 955.909, 1380.801, 975.931]], "bbox": [1130.377, 955.909, 371.147, 354.125], "ignore": 0, "iscrowd": 0, "area": 114500.101},
                {"id": 14, "image_id": 1, "category_id": 6, "segmentation": [[1140.377, 960.909, 1385.801, 980.931]], "bbox": [1140.377, 960.909, 373.147, 356.125], "ignore": 0, "iscrowd": 0, "area": 115000.101},
                {"id": 15, "image_id": 1, "category_id": 7, "segmentation": [[1150.377, 965.909, 1390.801, 985.931]], "bbox": [1150.377, 965.909, 375.147, 358.125], "ignore": 0, "iscrowd": 0, "area": 115500.101}
            ],
            "info": {
                "year": 2024, "version": "1.0", "description": "", "contributor": "Label Studio", "url": "", "date_created": "2024-03-19 21:53:23.756107"
            }
        }

        # # Mock the open method to read result.json content
        mock_open().read.return_value = json.dumps(result_json)

        # Create a temporary directory for test outputs
        temp_dir = '/tmp/data'
        os.makedirs(temp_dir, exist_ok=True)
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Run the function
        output = data_preparation(s3_dataset_path='dataset/custom-dataset',
                                  batch_size=4,
                                  train_loader_path=os.path.join(temp_dir, 'train_loader.pth'),
                                  test_loader_path=os.path.join(temp_dir, 'test_loader.pth'))

        # Verify
        mock_s3.download_file.assert_any_call('dataset/custom-dataset/images/028d2ad5-IMG_4850.jpg', f'{images_dir}/028d2ad5-IMG_4850.jpg')
        mock_s3.download_file.assert_any_call('dataset/custom-dataset/images/04180cf3-IMG_4875.jpg', f'{images_dir}/04180cf3-IMG_4875.jpg')
        self.assertEqual(output[0], 8)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
