import numpy as np

frames_stas = np.loadtxt('../data/hmdb51_frames_stats.txt')
hist = np.histogram(frames_stas, bins=int(frames_stas.max()), range=(0, frames_stas.max()))
np.savetxt('../data/hmdb51_frames_dist.txt', hist[0], fmt='%d')

print(frames_stas.min(), frames_stas.max(), frames_stas.mean(), frames_stas.std())