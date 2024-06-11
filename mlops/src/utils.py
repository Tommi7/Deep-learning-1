import tensorflow as tf # type: ignore
import numpy as np

def non_max_suppression(predictions, iou_threshold, confidence_threshold):
    """
    Apply non-maximum suppression to filter out overlapping bounding boxes.
    Args:
        predictions (Tensor): Predicted bounding boxes with format [x1, y1, x2, y2, score, class].
        iou_threshold (float): IoU threshold for suppression.
        confidence_threshold (float): Confidence threshold for filtering.
    Returns:
        Tensor: Filtered bounding boxes.
    """
    """
    Apply non-maximum suppression to filter out overlapping bounding boxes.
    Args:
        predictions (tf.Tensor): Predicted bounding boxes with format [x1, y1, x2, y2, score, class].
        iou_threshold (float): IoU threshold for suppression.
        confidence_threshold (float): Confidence threshold for filtering.
    Returns:
        tf.Tensor: Filtered bounding boxes.
    """
    # Filter out low-confidence boxes
    mask = predictions[:, 4] > confidence_threshold
    predictions = tf.boolean_mask(predictions, mask)

    # If no boxes remain, return an empty tensor
    if tf.size(predictions) == 0:
        return tf.constant([])

    # Sort the boxes by confidence score in descending order
    sorted_indices = tf.argsort(predictions[:, 4], direction='DESCENDING')
    predictions = tf.gather(predictions, sorted_indices)

    filtered_boxes = []

    while tf.size(predictions) > 0:
        # Select the box with the highest score and remove it from predictions
        chosen_box = predictions[0]
        filtered_boxes.append(chosen_box)
        predictions = predictions[1:]

        # Compute IoU of the chosen box with all remaining boxes
        ious = tf.map_fn(
            lambda box: calculate_iou(chosen_box[:4], box[:4]), 
            predictions, 
            dtype=tf.float32
        )

        # Keep boxes with IoU less than the threshold
        mask = ious < iou_threshold
        predictions = tf.boolean_mask(predictions, mask)

    return tf.stack(filtered_boxes)

def calculate_iou(prediction_boxes, target_boxes):
    """
    Calculate Intersection over Union(IoU) between two bounding boxes.

    Parameters:
    - box1: Tuple of 4 integers representing the coordinates of the first box.
    - box2: Tuple of 4 integers representing the coordinates of the second box.

    Returns:
    - iou: Intersection over Union value.
    """

    IoUs = []
    print(f'loss function gives: {prediction_boxes.shape}')
    prediction_boxes = prediction_boxes.numpy()
    print(f'numpy makes it this shape: {prediction_boxes.shape}')
    target_boxes = target_boxes.numpy()
    for prediction_row, target_row in zip(prediction_boxes, target_boxes):
        print(f'a row has {prediction_row.shape[0]} cells with bboxes')
        for pred_bboxes, target_bboxes in zip(prediction_row, target_row):
            print(f'a cell has {pred_bboxes.shape[0]} bboxes')
            for box1, box2, in zip(pred_bboxes, target_bboxes):
                print(f'a bbox has {box1.shape[0]} values')
                # Extract coordinates
                x1_1, y1_1, x2_1, y2_1 = box1
                x1_2, y1_2, x2_2, y2_2 = box2

                # Calculate intersection area
                x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
                y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
                intersection_area = x_intersection * y_intersection

                # Calculate areas of the boxes
                area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

                # Calculate Union area
                union_area = area_box1 + area_box2 - intersection_area

                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0.0
                
                IoUs.append(iou)
    tf.convert_to_tensor(IoUs, dtype=np.float32)
    print(IoUs.shape)
    tf.reshape(IoUs, (9, 16, 3))
    print(IoUs)
    return IoUs

if __name__ == '__main__':
    predictions = tf.constant([
    [100, 100, 210, 210, 0.9, 1],
    [105, 105, 215, 215, 0.8, 1],
    [250, 250, 400, 400, 0.7, 1]
    ], dtype=tf.float32)
    iou_threshold = 0.5
    confidence_threshold = 0.6

    filtered_boxes = non_max_suppression(predictions, iou_threshold, confidence_threshold)
    print(filtered_boxes)