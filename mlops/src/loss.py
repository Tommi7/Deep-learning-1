import tensorflow as tf # type: ignore
from utils import calculate_iou

def yolo_loss(predictions, targets, W=16, H=9, B=3, lambda_coord=5, lambda_noobj=0.5):
    """
    Compute the YOLOv1 loss.
    Args:
        predictions (tf.Tensor): Predicted output of shape (batch_size, W, H, B*5).
        targets (tf.Tensor): Ground truth targets of shape (batch_size, W, H, B*5).
        W (int): Number of horizontal grid cells. Default is 16.
        H (int): Number of vertical grid cells. Default is 9.
        B (int): Number of bounding boxes per grid cell. Default is 3.
        lambda_coord (float): Weight for coordinate loss.
        lambda_noobj (float): Weight for no-object confidence loss.
    Returns:
        tf.Tensor: Loss value.
    """
    print(predictions)
    # Separate the predictions into components
    pred_confidence = predictions[..., 4:5]
    pred_boxes = predictions[..., :4]

    # Separate the targets into components 
    target_confidence = targets[..., 4:5]
    target_boxes = targets[..., :4]

    # Compute IoU for the predicted and target boxes
    iou_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def compute_iou_loop_body(i, iou_scores):
        iou = calculate_iou(pred_boxes[i], target_boxes[i])
        iou_scores = iou_scores.write(i, iou)
        return i + 1, iou_scores

    _, iou_scores = tf.while_loop(
        lambda i, *_: i < tf.shape(pred_boxes)[0],
        compute_iou_loop_body,
        [0, iou_scores]
    )
    
    iou_scores = iou_scores.stack()
    best_iou = tf.reduce_max(iou_scores, axis=-1, keepdims=True)

    # Create masks
    obj_mask = target_confidence
    noobj_mask = 1 - obj_mask

    # Coordinate loss
    box_xy = pred_boxes[..., :2]
    box_wh = pred_boxes[..., 2:4]
    target_xy = target_boxes[..., :2]
    target_wh = target_boxes[..., 2:4]

    coord_loss = tf.reduce_sum(
        lambda_coord * obj_mask * (tf.square(target_xy - box_xy) + tf.square(tf.sqrt(target_wh) - tf.sqrt(box_wh)))
    )

    # Confidence loss
    obj_conf_loss = tf.reduce_sum(
        obj_mask * tf.square(pred_confidence - best_iou)
    )
    noobj_conf_loss = tf.reduce_sum(
        lambda_noobj * noobj_mask * tf.square(pred_confidence)
    )

    total_loss = coord_loss + obj_conf_loss + noobj_conf_loss

    return total_loss



