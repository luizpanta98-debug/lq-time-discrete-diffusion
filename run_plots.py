NO_DISPLAY_IMAGES = 5
torch_image_batch = torch.stack([torch_image] * NO_DISPLAY_IMAGES)
t = torch.linspace(0, diffusion_model.timesteps - 1, NO_DISPLAY_IMAGES).long()
noisy_image_batch, _ = diffusion_model.forward(torch_image_batch, t, device)

# Adjusted figsize to better match a 1x5 horizontal row
f, ax = plt.subplots(1, NO_DISPLAY_IMAGES, figsize=(25, 5))

# Pull the subplots tightly together (wspace controls width spacing)
plt.subplots_adjust(wspace=0.05) 

for idx, image in enumerate(noisy_image_batch):
    ax[idx].imshow(reverse_transform(image))
    ax[idx].set_title(f"Iteration: {t[idx].item()}", fontsize=20)
    
    # Removes axes ticks and borders to bring images flush
    ax[idx].axis('off') 

plt.show()