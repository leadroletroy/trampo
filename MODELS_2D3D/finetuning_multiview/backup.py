def predict_multiview_with_grad(detector, pose_estimator, images, bbox_thr=0.3, nms_thr=0.3):
    """
    Multi-view prediction with frozen detector but trainable pose estimator.
    
    Args:
        detector: mmdet model (frozen)
        pose_estimator: RTMPose or ViTPose model (trainable)
        images: torch.Tensor of shape (V, C, H, W) in [0,1]
    Returns:
        keypoints_per_view: list[(num_persons, num_kpts, 3)] with x,y,score
        bboxes_per_view:   list[(num_persons, 4)]
    """
    # ---- Forcer le modèle sur CPU ---- Switch to GPU when stable
    device = torch.device("cuda:0")
    pose_estimator = pose_estimator.to(device)

    B, V, C, H, W = images.size()
    keypoints_per_view = {b:{v:[] for v in range(V)} for b in range(B)}

    # Convert to numpy for detector
    imgs_np = images.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    imgs_np = imgs_np.astype(np.uint8)

    for b in range(B):
        for v in range(V):
            img = imgs_np[b, v]

            # -------- Stage 1: Person detection (frozen) --------
            detector.eval()
            for p in detector.parameters():
                p.requires_grad = False
            det_result = inference_detector(detector, img)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = pred_instance.bboxes
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > bbox_thr)]

            # -------- Stage 2: Pose Estimation (trainable) --------
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox.tolist())
                crop = img[y1:y2, x1:x2]

                """ im_rect = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 3)
                plt.figure()
                plt.imshow(im_rect)
                plt.show() """

                # --- Preprocessing
                model_input_size = (256, 192)  # match your model's expected input size (W,H)
                crop_resized, scale, pads = resize_and_pad_keep_aspect(crop, model_input_size)

                crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
                crop_tensor = crop_tensor.unsqueeze(0).to(device)
                
                # --- Forward pass
                pose_estimator.eval()  # make sure it’s in train mode
                output_x, output_y = pose_estimator(crop_tensor, None, mode='tensor')  # raw tensor output
                
                # --- Decoding 
                keypoints, scores = decode_simcc(output_x, output_y, input_size=(256, 192), T=0.1)
                keypoints = keypoints.squeeze()
                scores = scores.squeeze()

                #show_keypoints_on_crop(crop_tensor, keypoints)
                origin_bbox = torch.tensor([x1, y1], dtype=torch.float32, device=device)
                pads = torch.tensor(pads, dtype=torch.int16, device=device)

                keypoints_bbox = map_keypoints_to_bbox(keypoints, scale, pads)
                keypoints_img = keypoints_bbox + origin_bbox

                keypoints_per_view[b][v].append((keypoints_img, scores))

    return keypoints_per_view



def predict_multiview_with_grad(detector, pose_estimator, images, bbox_thr=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_estimator = pose_estimator.to(device)

    B, V, C, H, W = images.size()

    # --- Convert to numpy for detection
    imgs_np = (images.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)

    # --- Run detection first to know how many persons per view
    all_bboxes = [[None for _ in range(V)] for _ in range(B)]
    max_persons = 0

    detector.eval()
    for b in range(B):
        for v in range(V):
            img = imgs_np[b, v]
            det_result = inference_detector(detector, img)
            pred_instance = det_result.pred_instances.cpu().numpy()
            bboxes = pred_instance.bboxes[
                np.logical_and(pred_instance.labels == 0, pred_instance.scores > bbox_thr)
            ]
            all_bboxes[b][v] = bboxes
            max_persons = max(max_persons, len(bboxes))

    # --- Prepare tensor for results
    num_kpts = pose_estimator.head.num_keypoints  # or set manually (e.g. 17)
    keypoints_all = torch.full(
        (B, V, max_persons, num_kpts, 3),
        torch.nan,
        dtype=torch.float32,
        device=device
    )

    # --- Pose estimation
    pose_estimator.train()  # IMPORTANT for finetuning

    for b in range(B):
        for v in range(V):
            img = imgs_np[b, v]
            bboxes = all_bboxes[b][v]
            for p_idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, bbox.tolist())
                crop = img[y1:y2, x1:x2]

                # --- Preprocessing
                model_input_size = (256, 192)
                crop_resized, scale, pads = resize_and_pad_keep_aspect(crop, model_input_size)
                crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
                crop_tensor = crop_tensor.unsqueeze(0).to(device)

                # --- Forward pass (with grad)
                output_x, output_y = pose_estimator(crop_tensor, None, mode='tensor')

                # --- Decode
                keypoints, scores = decode_simcc(output_x, output_y, input_size=(256, 192), T=0.1)
                keypoints = keypoints.squeeze(0)
                scores = scores.squeeze(0)

                # --- Map back to full image coords
                origin_bbox = torch.tensor([x1, y1], dtype=torch.float32, device=device)
                pads = torch.tensor(pads, dtype=torch.float32, device=device)
                keypoints_bbox = map_keypoints_to_bbox(keypoints, scale, pads)
                keypoints_img = keypoints_bbox + origin_bbox

                # --- Store directly
                data = torch.cat([keypoints_img, scores.unsqueeze(-1)], dim=1)
                keypoints_all[b, v, p_idx] = data

    return keypoints_all


def persons_cominations(keypoints_per_view):
    n_cams = keypoints_per_view.size()[1]
    kpts = keypoints_per_view[..., 0]  # shape (B, V, N_local, K)
    valid = ~torch.isnan(kpts)
    nb_persons_per_cam = valid.any(dim=-1).sum(dim=-1)  # shape (B,V)     

    # persons combinations
    id_no_detect = [i for i, x in enumerate(nb_persons_per_cam) if x == 0]  # ids of cameras that have not detected any person
    nb_persons_per_cam = [x if x != 0 else 1 for x in nb_persons_per_cam] # temporarily replace persons count by 1 when no detection
    range_persons_per_cam = [range(nb_persons_per_cam[c]) for c in range(n_cams)] 
    personsIDs_comb = np.array(list(it.product(*range_persons_per_cam)), float) # all possible combinations of persons' ids
    personsIDs_comb[:,id_no_detect] = np.nan # -1 = persons' ids when no person detected
    
    return personsIDs_comb

def persons_cominations(keypoints_per_view):
    n_cams = len(keypoints_per_view.values())
    nb_persons_per_cam = [len(el) for el in keypoints_per_view.values()]

    # persons combinations
    id_no_detect = [i for i, x in enumerate(nb_persons_per_cam) if x == 0]  # ids of cameras that have not detected any person
    nb_persons_per_cam = [x if x != 0 else 1 for x in nb_persons_per_cam] # temporarily replace persons count by 1 when no detection
    range_persons_per_cam = [range(nb_persons_per_cam[c]) for c in range(n_cams)] 
    personsIDs_comb = np.array(list(it.product(*range_persons_per_cam)), float) # all possible combinations of persons' ids
    personsIDs_comb[:,id_no_detect] = np.nan # -1 = persons' ids when no person detected
    
    return personsIDs_comb

def get_persons(keypoints, combination):
    """
    Return the (keypoints, scores) arrays of selected persons in a given combination
    Args:
        keypoints: (8, 2, N, 17) list of detections by cameras
        combination: (8) list of selected detections
    Return:
        keypoints: (8, 17, 2) list of selected detections' keypoints
        scores: (8, 17) list of selected detections' confidence scores
    """
    selected_points, selected_scores = [], []
    for cam_idx, person_idx in enumerate(combination):
        if not math.isnan(person_idx):
            selected_points.append(keypoints[cam_idx][int(person_idx)][0].squeeze().to(dtype=torch.float32, device='cuda:0'))
        else:
            selected_points.append(torch.zeros((17, 2), device='cuda:0'))
        #selected_scores.append(keypoints[cam_idx][1][int(person_idx)].squeeze())

    return torch.stack(selected_points) #, np.array(selected_scores)