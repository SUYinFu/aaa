import base64  
import io  
import torch  
import numpy as np  
import torch.nn.functional as F  
import onnxruntime as rt  
import cv2  

def keepratio_resize(img):  
    cur_ratio = img.shape[1] / float(img.shape[0])  
    mask_height = 32  
    mask_width = 804  
    if cur_ratio > float(mask_width) / mask_height:  
        cur_target_height = mask_height  
        cur_target_width = mask_width  
    else:  
        cur_target_height = mask_height  
        cur_target_width = int(mask_height * cur_ratio)  
    img = cv2.resize(img, (cur_target_width, cur_target_height))  
    mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)  
    mask[:img.shape[0], :img.shape[1], :] = img  
    img = mask  
    return img  

def load_label_mapping(file_path):  
    labelMapping = dict()  
    with open(file_path, 'r', encoding='utf-8') as f:  
        lines = f.readlines()  
        cnt = 2  
        for line in lines:  
            line = line.strip('\n')  
            labelMapping[cnt] = line  
            cnt += 1  
    return labelMapping  

def recognize_image(base64_image, model_path, vocab_path):  
    # Decode base64 image  
    img_bytes = base64.b64decode(base64_image)  
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)  
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  

    # Preprocess image  
    img = keepratio_resize(img)  
    img = torch.FloatTensor(img)  

    chunk_img = []  
    for i in range(3):  
        left = (300 - 48) * i  
        chunk_img.append(img[:, left:left + 300, :])  
    merge_img = torch.cat(chunk_img, 0)  
    data = merge_img.view(3, 32, 300, 3) / 255.  
    data = data.permute(0, 3, 1, 2).cuda()  
    input_data = data.cpu().numpy()  

    # Load model and perform inference  
    sess = rt.InferenceSession(model_path)  
    input_name = sess.get_inputs()[0].name  
    output_name = sess.get_outputs()[0].name  
    res = sess.run([output_name], {input_name: input_data})  
    outprobs = F.softmax(torch.tensor(res[0]), dim=-1)  
    preds = torch.argmax(outprobs, -1)  

    # Load label mapping and decode predictions  
    labelMapping = load_label_mapping(vocab_path)  
    batchSize, length = preds.shape  
    final_str_list = []  
    for i in range(batchSize):  
        pred_idx = preds[i].cpu().data.tolist()  
        last_p = 0  
        str_pred = []  
        for p in pred_idx:  
            if p != last_p and p != 0:  
                str_pred.append(labelMapping[p])  
            last_p = p  
        final_str = ''.join(str_pred)  
        final_str_list.append(final_str)  

    return final_str_list  

# Example usage  
# base64_image = 'base64encodedimage'  
# model_path = './onnx/model.onnx'  
# vocab_path = 'vocab.txt'  
# result = recognize_image(base64_image, model_path, vocab_path)  
# print(result)