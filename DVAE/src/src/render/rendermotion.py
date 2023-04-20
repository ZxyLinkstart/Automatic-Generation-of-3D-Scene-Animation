import os
import sys
sys.path.append("/data1/zxy/ACTOR/")
import numpy as np
import imageio
import argparse
from tqdm import tqdm
from src.render.renderer import get_renderer



def get_rotation(theta=np.pi/3):
    import src.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()


def render_video(meshes, key, action, renderer, savepath, background, cam=(0.75, 0.75, 0, 0.10), color=[1.0, 1.0, 0.9]):
    writer = imageio.get_writer(savepath, fps=30)
    # center the first frame
    meshes = meshes - meshes[0].mean(axis=0)
    # matrix = get_rotation(theta=np.pi/4)
    # meshes = meshes[45:]
    # meshes = np.einsum("ij,lki->lkj", matrix, meshes)
    imgs = []

    frame = 0
    for mesh in tqdm(meshes, desc=f"Visualize {key}, action {action}"):
        if not os.path.exists("/data1/zxy/ACTOR/mesh/{}_{}".format(key,action)):
            os.mkdir("/data1/zxy/ACTOR/mesh/{}_{}".format(key,action))
            print("make dir:","/data1/zxy/ACTOR/mesh/{}_{}".format(key,action))
        path = '/data1/zxy/ACTOR/mesh/{}_{}/{:0>4d}.obj'.format(key,action,frame)
        img = renderer.render(background, mesh, cam, color=color,mesh_filename = path)
        imgs.append(img)
        frame += 1
        # show(img)
    
    imgs = np.array(imgs)
    masks = ~(imgs/255. > 0.96).all(-1)

    coords = np.argwhere(masks.sum(axis=0))
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    for cimg in imgs[:, y1:y2, x1:x2]:
        writer.append_data(cimg)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    opt = parser.parse_args()

    filename = opt.filename  # /data1/zxy/ACTOR/pretrained_models/humanact12/generation.npy
    savefolder = os.path.splitext(filename)[0]
    os.makedirs(savefolder, exist_ok=True)
    print("save_folder",savefolder)
    print("filename",filename)

    # 加载模型
    output = np.load(filename)

    if output.shape[0] == 3:
        visualization, generation, reconstruction = output
        output = {"visualization": visualization,
                  "generation": generation,
                  "reconstruction": reconstruction}
    else:
        # output = {f"generation_{key}": output[key] for key in range(2)} #  len(output))}
        # output = {f"generation_{key}": output[key] for key in range(len(output))}
        output = {f"generation_{key}": output[key] for key in range(len(output))}

    width = 1024
    height = 1024

    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height)

    # if duration mode, put back durations
    # if output["generation_3"].shape[-1] == 100:
    #     output["generation_0"] = output["generation_0"][:, :, :, :40]
    #     output["generation_1"] = output["generation_1"][:, :, :, :60]
    #     output["generation_2"] = output["generation_2"][:, :, :, :80]
    #     output["generation_3"] = output["generation_3"][:, :, :, :100]
    # elif output["generation_3"].shape[-1] == 160:
    #     print("160 mode")
    #     output["generation_0"] = output["generation_0"][:, :, :, :100]
    #     output["generation_1"] = output["generation_1"][:, :, :, :120]
    #     output["generation_2"] = output["generation_2"][:, :, :, :140]
    #     output["generation_3"] = output["generation_3"][:, :, :, :160]

    # if str(action) == str(1) and str(key) == "generation_4":
    
    for key in output:
        print("key:",key)
        vidmeshes = output[key]
        for action in range(1):
            meshes = vidmeshes.transpose(2, 0, 1)
            # meshes = vidmeshes[action].transpose(2, 0, 1)
            path = os.path.join(savefolder, "action{}_{}.mp4".format(action, key))
            render_video(meshes, key, action, renderer, path, background)


if __name__ == "__main__":
    main()
