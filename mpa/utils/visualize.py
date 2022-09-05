import numpy as np
import cv2

def _visualize_det_img(imgs, img_metas, labels, bboxes):
    for i, img in enumerate(imgs):
        np_img = np.transpose(np.array(img.detach().cpu()*255, dtype=np.uint8), (1,2,0))
        print_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        for b in bboxes[i]:
            print(b)
            np_b = np.array(b.detach().cpu().numpy())
            cv2.rectangle(print_img, (int(np_b[0]), int(np_b[1])), (int(np_b[2]), int(np_b[3])), (255,0,0), 2)
        
        cv2.imwrite('temp.jpg', print_img)
        raise
    
    raise