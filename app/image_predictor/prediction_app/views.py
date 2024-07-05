# prediction_app/views.py
from django.http import JsonResponse
from django.shortcuts import render
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from .image_utils import prepare_image, create_payload  # Assuming these handle image preprocessing and payload creation
from .utils import select_top_predictions_per_group, COCO_CATEGORIES  # Assuming COCO_CATEGORIES is a list of category names
import requests
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def image_upload_view(request):
    # This view is now streamlined to return JSON responses for SPA functionality.
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            image_path = fs.path(filename)

            # Prepare and send payload to the model server
            try:
                image_data = prepare_image(image_path)
                payload = create_payload(image_data)
                headers = {"Authorization": f"Bearer {settings.TOKEN}", "Content-Type": "application/json"}
                model_response = requests.post(settings.MODEL_SERVER_URL, json=payload, headers=headers, verify=settings.VERIFY_SSL)

                if model_response.status_code == 200:
                    # Process model response
                    predictions_response = model_response.json()
                    raw_predictions = predictions_response['outputs'][0]['data']
                    top_predictions = select_top_predictions_per_group(raw_predictions, n=8)
                    
                    # Respond with the uploaded image URL and the top predictions
                    return JsonResponse({
                        'uploaded_file_url': uploaded_file_url,
                        'predictions': top_predictions,
                    })
                else:
                    logger.error("Model server response error. Status code: %s, Response: %s", model_response.status_code, model_response.text)
                    return JsonResponse({'error': 'Failed to get predictions from the model server.'}, status=500)
            except Exception as e:
                logger.error("Exception occurred: %s", str(e), exc_info=True)
                return JsonResponse({'error': 'An internal error occurred.'}, status=500)
        else:
            return JsonResponse({'error': 'Invalid form submission.'}, status=400)
    
    # For GET requests or non-form submission POST requests, render the SPA template.
    return render(request, 'prediction_app/index.html', {'form': ImageUploadForm()})
