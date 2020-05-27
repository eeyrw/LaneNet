import argparse
from config import *

from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
from PIL import Image
from torchvision import transforms
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/0ac20953-2830f7c0.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, default="exp0.pth", help="Path to model weights")
    parser.add_argument("--band_width", '-b', type=float, default=1.5, help="Value of delta_v")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args

def resizeAndCropToTargetSize(img, width, height):
    rawW, rawH = img.size
    rawAspectRatio = rawW/rawH
    wantedAspectRatio = width/height
    if rawAspectRatio > wantedAspectRatio:
        scaleFactor = height/rawH
        widthBeforeCrop = int(rawW*scaleFactor)
        return img.resize((widthBeforeCrop, height), Image.BILINEAR). \
            crop(((widthBeforeCrop-width)//2, 0,
                  (widthBeforeCrop-width)//2+width, height))
    else:
        scaleFactor = width/rawW
        heightBeforeCrop = int(rawH*scaleFactor)
        return img.resize((width, heightBeforeCrop), Image.BILINEAR). \
            crop((0, (heightBeforeCrop-height)//2, width,
                  (heightBeforeCrop-height)//2+height))

def visualizeEmbedding(embedding):
    # shape:(1, 4, 288, 800)
    embedding= embedding[0]
    maxV = embedding.max()
    minV = embedding.min()
    embedding = (embedding-minV)/(maxV-minV)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='float')
    darkGroundImage = np.zeros((embedding.shape[1],embedding.shape[2],3),dtype='uint8') # HWC
    for i, embeddingLayer in enumerate(embedding):
        #colorMap = [(color*i/256).astype(np.uint8) for i in range(256)]
        layer = cv2.applyColorMap((embeddingLayer*255).astype(np.uint8),cv2.COLORMAP_JET)
        #darkGroundImage = cv2.addWeighted(src1=darkGroundImage, alpha=0.3, src2=layer, beta=1., gamma=0.)
        cv2.imwrite("demo/embedding_%d.png"%i, layer)        
    

def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

    _set = "IMAGENET"
    mean = IMG_MEAN[_set]
    std = IMG_STD[_set]
    transform_x = Compose(transforms.ToTensor(), transforms.Normalize(mean=mean, std=std))

    img = Image.open(img_path).convert('RGB')
    img = resizeAndCropToTargetSize(img,800,288)
    x = transform_x(img)
    x.unsqueeze_(0)

    net = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
    save_dict = torch.load(weight_path, map_location='cpu')['net']
    net.load_state_dict(save_dict)
    net.eval()

    output = net(x)
    embedding = output['embedding']
    embedding = embedding.detach().cpu().numpy()
    visualizeEmbedding(embedding)
    embedding = np.transpose(embedding[0], (1, 2, 0))
    binary_seg = output['binary_seg']
    bin_seg_prob = binary_seg.detach().cpu().numpy()
    bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]
    cv2.imwrite("demo/demo_seg.png", bin_seg_pred*255)

    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
    seg_img = np.zeros_like(img)
    lane_seg_img = embedding_post_process(embedding, bin_seg_pred, args.band_width, 4)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        if lane_idx==0:
            continue
        seg_img[lane_seg_img == lane_idx] = color[i-1]
    img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)

    cv2.imwrite("demo/demo_result.jpg", img)

    if args.visualize:
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
