from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import imageio

# Configuration
input_folder = '/Users/shrabin/Desktop/PIV/input_e_min0 copy'
output_folder = '/Users/shrabin/Desktop/PIV/outputs_combined'
os.makedirs(output_folder, exist_ok=True)


winsize = 24
searchsize = 30
overlap = 12

def normalized_cross_correlation(win_a, win_b):
    """
    Compute normalized cross-correlation between two windows.
    Output is bounded between -1 and 1.
    """
    a = win_a.astype(np.float32)
    b = win_b.astype(np.float32)

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)

    # Avoid divide by zero
    if std_a == 0 or std_b == 0:
        return np.array([[0]])

    a_norm = a - mean_a
    b_norm = b - mean_b

    # Pad b so it can slide over a
    from scipy.signal import correlate2d
    corr = correlate2d(a_norm, b_norm, mode='valid') / (std_a * std_b * a.size)

    return corr


tif_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))


for woo in range(2, 4):

    frame_skip = woo
    dt = 1/2100 * woo


    frames_corr = []
    frames_vector = []

    fig_corr, ax_corr = plt.subplots(figsize=(8, 8), dpi=150)
    fig_vec, ax_vec = plt.subplots(figsize=(8, 8), dpi=150)

    # Process each pair
    for i in range(0, len(tif_files) - frame_skip, frame_skip):
        frame_a = np.flipud(tools.imread(tif_files[i]))
        frame_b = np.flipud(tools.imread(tif_files[i + frame_skip]))
        height, width = frame_a.shape
        nx = (width - winsize) // overlap + 1
        ny = (height - winsize) // overlap + 1
        
        # ---------- Video A: Correlation Coefficient Map ----------
        corr_map = np.zeros((ny, nx))
        for j in range(ny):
            for k in range(nx):
                x0 = k * overlap
                y0 = j * overlap
                win_a = frame_a[y0:y0+winsize, x0:x0+winsize]
                win_b = frame_b[y0:y0+searchsize, x0:x0+searchsize]
                if win_a.shape != (winsize, winsize) or win_b.shape != (searchsize, searchsize):
                    continue
                
                corr = normalized_cross_correlation(win_a, win_b)
                corr_map[j, k] = np.max(corr)

        ax_corr.clear()
        im = ax_corr.imshow(corr_map, cmap='gray', origin='lower')
        ax_corr.set_title(f'Correlation Peak - Frame {i}')

        if i == 0:
            fig_corr.colorbar(im, ax=ax_corr, label='Peak Correlation')

        fig_corr.canvas.draw()
        img_corr = np.frombuffer(fig_corr.canvas.tostring_rgb(), dtype='uint8')
        img_corr = img_corr.reshape(fig_corr.canvas.get_width_height()[::-1] + (3,))
        frames_corr.append(img_corr)

        # ---------- Video B: Vector Field with Speed Background ----------
        u0, v0, s2n = pyprocess.extended_search_area_piv(
            frame_a.astype(np.int32),
            frame_b.astype(np.int32),
            window_size=winsize,
            overlap=overlap,
            dt=dt,
            search_area_size=searchsize,
            sig2noise_method='peak2peak',
        )

        x, y = pyprocess.get_coordinates(frame_a.shape, search_area_size=searchsize, overlap=overlap)

        mask = validation.sig2noise_val(s2n, threshold=1.6)
 
        u2, v2 = filters.replace_outliers(u0, v0, mask, method='distance', max_iter=10, kernel_size=1)

        x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor= 1)

        speed = np.sqrt(u3**2 + v3**2)

        ax_vec.clear()
        c = ax_vec.pcolormesh(x, y, speed, shading='auto', cmap='turbo')  # background
        ax_vec.quiver(
            x, y, 
            np.ma.array(u3, mask=mask), 
            np.ma.array(v3, mask=mask), 
            scale= 1, width=0.0035, color='black'
        )
        ax_vec.set_title(f'Vector Field - Frame {i}')

        if i == 0:
            fig_vec.colorbar(c, ax=ax_vec, label='Velocity Magnitude')

        fig_vec.canvas.draw()
        img_vec = np.frombuffer(fig_vec.canvas.tostring_rgb(), dtype='uint8')
        img_vec = img_vec.reshape(fig_vec.canvas.get_width_height()[::-1] + (3,))
        frames_vector.append(img_vec)

    plt.close(fig_corr)
    plt.close(fig_vec)

    # ---------- Save both videos ----------
    video_path_corr = os.path.join(output_folder, f"test_{woo}video_A_correlation_map.mp4")
    video_path_vec = os.path.join(output_folder, f"test_{woo}video_B_vector_field.mp4")

    imageio.mimsave(video_path_corr, frames_corr, fps=7.5, format='ffmpeg')
    imageio.mimsave(video_path_vec, frames_vector, fps=7.5, format='ffmpeg')

    print(f"✅ Correlation map video saved to: {video_path_corr}")
    print(f"✅ Vector + magnitude video saved to: {video_path_vec}")
