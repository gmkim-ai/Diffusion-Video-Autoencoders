from templates import *
from templates_cls import *
from torchvision.utils import *
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from losses.clip_loss import CLIPLoss
import dlib
import numpy as np
import skimage.io as io
from PIL import Image
import scipy
from scipy.ndimage import gaussian_filter1d
import PIL
import argparse
import imageio

def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def paste_image(coeffs, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image

def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)

def get_landmark(filepath, predictor, detector=None, fa=None):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    if fa is not None:
        image = io.imread(filepath)
        lms, _, bboxes = fa.get_landmarks(image, return_bboxes=True)
        if len(lms) == 0:
            return None
        return lms[0]

    if detector is None:
        detector = dlib.get_frontal_face_detector()
    if isinstance(filepath, PIL.Image.Image):
        img = np.array(filepath)
    else:
        img = dlib.load_rgb_image(filepath)
    dets = detector(img)

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        break
    else:
        return None
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm

def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None):
    lm = get_landmark(filepath, predictor, detector, fa)
    if lm is None:
        raise Exception(f'Did not detect any faces in image: {filepath}')
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y
    
def crop_image(filepath, output_size, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    else:
        img = PIL.Image.open(filepath)
    transform_size = output_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)
        quad -= crop[0:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4, help="batch size for editing process")
parser.add_argument("--T", type=int, default=1000, help="total diffusion step")
parser.add_argument("--n_train_step", type=int, default=5, help="total step for computing CLIP loss in diffusion process")
parser.add_argument("--max_num", type=int, default=None, help="maximum number of frames to edit, 1~max_num")
parser.add_argument("--attribute", type=str, default="beards", required=True, help="attribute name for making folders")
parser.add_argument("--src_txt", type=str, default="face", required=True, help="neutral text for editing")
parser.add_argument("--trg_txt", type=str, default="face with beard", required=True, help="target text for editing")
parser.add_argument("--lr", type=float, default=0.0020, help="learning rate")
parser.add_argument("--scale", type=float, default=1.0, help="scale factor of editing step")
parser.add_argument("--clip_loss_w", type=float, default=3, help="weight for clip loss")
parser.add_argument("--id_loss_w", type=int, default=5, help="weight for id loss")
parser.add_argument("--l1_loss_w", type=int, default=5, help="weight for l1 loss")
parser.add_argument("--video_path", type=str, default="sample_video", required=True, help="path to video frames")
args = parser.parse_args()

gpus = [0]
device = 'cuda:0'

conf = diffusion_video_autoencoder(gpus)
state = torch.load(f'checkpoints/{conf.name}/epoch=51-step=1000000.ckpt', map_location='cpu')     

# print(conf.name)
model = LitModel(conf)
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

sampler = model.conf._make_diffusion_conf(args.n_train_step).make_sampler()

log_dir = f'editing_CLIP/{args.video_path}'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(os.path.join(log_dir, "crop")):
    os.mkdir(os.path.join(log_dir, "crop"))
if not os.path.exists(os.path.join(log_dir, "recon")):
    os.mkdir(os.path.join(log_dir, "recon"))

predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

print("Loading losses")
clip_loss_func = CLIPLoss(
    device,
    lambda_direction=1,
    lambda_patch=0,
    lambda_global=0,
    lambda_manifold=0,
    lambda_texture=0,
    clip_model='ViT-B/16')

images = []
dir_name = args.video_path
for fname in sorted(os.listdir(dir_name)):
    path = os.path.join(dir_name, fname)
    fname = fname.split('.')[0]
    images.append((fname, path))
cs, xs, ys = [], [], []
for _, path in images:
    c, x, y = compute_transform(path, predictor, detector=detector, scale=1.0)
    cs.append(c)
    xs.append(x)
    ys.append(y)
cs = np.stack(cs)
xs = np.stack(xs)
ys = np.stack(ys)
cs = gaussian_filter1d(cs, sigma=1.0, axis=0)
xs = gaussian_filter1d(xs, sigma=3.0, axis=0)
ys = gaussian_filter1d(ys, sigma=3.0, axis=0)
quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
quads = list(quads)
orig_images = []
for quad, (_, path) in tqdm(zip(quads, images), total=len(quads)):
    crop = crop_image(path, 1024, quad.copy())
    crop = crop.convert('RGB')
    crop.save(f'{log_dir}/crop/%s.jpg' % path.split('/')[-1].split('.')[0])#, quality=100, subsampling=0)
    orig_image = Image.open(path)
    orig_images.append(orig_image)
image_size = 256
inverse_transforms = [
    calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
    for quad in quads]            

data = ImageDataset(f'{log_dir}/crop', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False, sort_names=True, max_num=args.max_num)
dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, drop_last=False)

conds = []
with torch.no_grad():
    for batch in dataloader:
        imgs = batch['img']
        indices = batch['index']

        cond = model.ema_model.encoder.id_forward(imgs.to(device))
        conds.append(cond)

cond = torch.cat(conds, dim=0)
avg_cond = torch.mean(cond, dim=0, keepdim=True)

with torch.no_grad():
    start_img = Image.open(f'{log_dir}/crop/%s.jpg' % images[0][1].split('/')[-1].split('.')[0])
    start_img = start_img.convert('RGB')
    if data.transform is not None:
        start_img = data.transform(start_img)
    start_cond = model.ema_model.encoder.id_forward(start_img.unsqueeze(0).to(device))  #[1, 512]

    start_latent = model.ema_model.encoder.forward_with_id(start_cond, start_img.unsqueeze(0).to(device))
    out = sampler.ddim_reverse_sample_loop(model.ema_model,
                                        start_img.unsqueeze(0).to(device),
                                        model_kwargs={'cond': start_latent, 'cond_other': None})

    forward_imgs = out['sample_t'][::-1] 
    forward_imgs.append(start_img.unsqueeze(0).to(device)) 
    forward_imgs = torch.cat(forward_imgs).detach()
    forward_recon_imgs = []
    
    start_recon_imgs = forward_imgs[0].unsqueeze(0)
    indices = list(range(sampler.num_timesteps))[::-1]   # [4, 3, 2, 1, 0]
    for i in indices:
        forward_recon_imgs.append(start_recon_imgs)
        t = torch.tensor([i], device=device)
        out = sampler.ddim_sample(
                model.ema_model,
                start_recon_imgs,
                t,
                model_kwargs={'cond': start_latent, 'cond_other': None})
        start_recon_imgs = out["sample"]
    forward_recon_imgs = torch.cat(forward_recon_imgs).detach()
    forward_recon_imgs = torch.cat([forward_recon_imgs, start_recon_imgs], axis=0) # [800, 600, 400, 200, 0, -1]
    start_mask = model.ema_model.encoder.face_mask(start_img.unsqueeze(0).to(device))
    
    for batch in dataloader:
        imgs = batch['img']
        batch_indices = batch['index']
        
        avg_latent = model.ema_model.encoder.forward_with_id(avg_cond.expand(len(imgs), -1), imgs.to(device))
        avg_xT = model.encode_stochastic(imgs.to(device), avg_latent, T=args.T)
        avg_img = model.render(avg_xT, avg_latent, T=args.T)

        ori = (imgs + 1) / 2
        for index in range(len(imgs)):
            file_name = data.paths[batch_indices[index]]
            save_image(ori[index], f'{log_dir}/recon/orig_{file_name}')
            save_image(avg_xT[index], f'{log_dir}/recon/avg_xT_{file_name}')
            save_image(avg_img[index], f'{log_dir}/recon/avg_recon_{file_name}')

if not os.path.exists(f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}'):
    os.mkdir(f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}')    

edit_cond = nn.Parameter(start_cond.clone())
optimizer = torch.optim.Adam([edit_cond], weight_decay=0, lr=args.lr)
clip_loss_func.target_direction = None

with torch.set_grad_enabled(True):
    for it in range(2000):
        cond = l2_norm(edit_cond)
        latent = model.ema_model.encoder.forward_with_id(cond, start_img.unsqueeze(0).to(device))
        optimizer.zero_grad()
        
        #Each t pair
        t = torch.tensor(indices, device=device)   # [4, 3, 2, 1, 0]
        out = sampler.ddim_sample(
                model.ema_model,
                forward_recon_imgs[:-1],   #[800, 600, 400, 200, 0]
                t,
                model_kwargs={'cond': latent, 'cond_other': None})
        backward_imgs = out["sample"]   # [600, 400, 200, 0, -1]

        loss_clip = (2 - clip_loss_func(forward_recon_imgs[1:], args.src_txt, backward_imgs, args.trg_txt)) / 2
        loss_clip = -torch.log(loss_clip)
        loss_id = torch.mean((1 - cond[0].dot(start_cond[0])))
        loss_l1 = nn.L1Loss()(forward_recon_imgs[1:] * start_mask, backward_imgs * start_mask)  

        loss = args.clip_loss_w * loss_clip + args.id_loss_w * loss_id + args.l1_loss_w * loss_l1
        loss.backward()
        print(f"CLIP opt: loss_clip: {loss_clip:.3f}, loss_id: {loss_id:.3f}, loss_l1: {loss_l1:.3f}")
        if (it + 1) % 100 == 0:
            save_image((make_grid(backward_imgs) + 1) / 2, f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}/backward_img_clip_{args.trg_txt.replace(" ", "_")}_{it+1}_ngen{args.n_train_step}.png')
        optimizer.step()

editing_step = edit_cond - start_cond

video_frames = []
with torch.no_grad():
    for batch in dataloader:
        imgs = batch['img']
        batch_indices = batch['index']

        avg_latent = model.ema_model.encoder.forward_with_id(avg_cond.expand(len(imgs), -1), imgs.to(device))
        avg_xT = model.encode_stochastic(imgs.to(device), avg_latent, T=args.T)
        mask = model.ema_model.encoder.face_mask(imgs.to(device), for_video=True)
        
        avg_cond2 = l2_norm(avg_cond + args.scale * editing_step)

        avg_latent2 = model.ema_model.encoder.forward_with_id(avg_cond2.expand(len(imgs), -1), imgs.to(device))

        avg_img = model.render(avg_xT, avg_latent2, T=args.T)

        for index in range(len(imgs)):
            file_name = data.paths[batch_indices[index]]
            # save_image(avg_img[index], f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}/avg_mani_{file_name}')
            paste_bg = avg_img[index].unsqueeze(0) * mask[index].unsqueeze(0) + ((imgs[index].to(device).unsqueeze(0) + 1) / 2 * (1 - mask[index].unsqueeze(0)))
            save_image(paste_bg[0], f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}/paste_avg_mani_{file_name}')
            paste_bg_crop = tensor2pil((paste_bg[0] * 2) - 1)
            paste_bg_pasted_image = paste_image(inverse_transforms[batch_indices[index]], paste_bg_crop, orig_images[batch_indices[index]])
            paste_bg_pasted_image = paste_bg_pasted_image.convert('RGB')
            video_frames.append(paste_bg_pasted_image)
            paste_bg_pasted_image.save(f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}/paste_final_avg_mani_{file_name}') 

imageio.mimwrite(f'{log_dir}/{args.attribute}_{args.lr:.4f}_id{args.id_loss_w}_l1{args.l1_loss_w}/out.mp4', video_frames, fps=20, output_params=['-vf', 'fps=20'])