import numpy as np
import cv2
import os
import random

def generate_puf_dataset(output_dir="dataset", num_images=100, img_size=(800, 200)):
    """
    Generate synthetic PUF (Physical Unclonable Function) images with random particles
    scattered across adhesive tape. These serve as master/authentic references.
    
    The particles simulate reflective beads embedded in the tape glue layer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        # 1. Initialize Canvas (Black Tape Base)
        # Start with dark base representing the tape substrate
        canvas = np.full(img_size, 30, dtype=np.float32) / 255.0  # ~30/255 intensity for black tape
        
        # Add subtle tape texture (simulating the fabric weave)
        texture = np.random.normal(0, 0.01, img_size).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (15, 15), 0)
        canvas = np.clip(canvas + texture, 0, 1)

        # 2. Generate random particles with truly uncorrelated distribution
        # This is key for PUF authenticity - particles should be impossible to replicate exactly
        num_particles = random.randint(800, 1500)  # High particle density
        
        for _ in range(num_particles):
            # Uniformly random position (not clustered - each PUF is unique)
            px = random.randint(0, img_size[1] - 1)
            py = random.randint(0, img_size[0] - 1)
            
            # Particle properties vary slightly for realism
            brightness = np.random.uniform(0.5, 1.0)  # Reflectivity
            sigma = np.random.uniform(0.5, 1.5)  # Particle size (in pixels)
            
            # Draw Gaussian blob to represent particle glow from flash
            k_size = max(3, int(sigma * 4) | 1)
            kernel = cv2.getGaussianKernel(k_size, sigma)
            kernel_2d = (kernel @ kernel.T) * brightness
            
            # Splat onto canvas with boundary handling
            h, w = kernel_2d.shape
            y1 = max(0, py - h//2)
            y2 = min(img_size[0], py + h//2 + 1)
            x1 = max(0, px - w//2)
            x2 = min(img_size[1], px + w//2 + 1)
            
            ky1 = h//2 - (py - y1)
            ky2 = h//2 + (y2 - py)
            kx1 = w//2 - (px - x1)
            kx2 = w//2 + (x2 - px)
            
            canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], 
                                               kernel_2d[ky1:ky2, kx1:kx2])

        # 3. Add realistic scanning artifacts (flash glare from phone camera)
        # Typically off-center due to phone flash position
        if random.random() > 0.3:  # 70% of scans have flash glare
            glare_x = random.randint(int(img_size[1]*0.3), int(img_size[1]*0.9))
            glare_y = random.randint(int(img_size[0]*0.4), int(img_size[0]*0.9))
            glare_intensity = random.uniform(0.15, 0.35)
            
            # Specular highlight (simulates camera flash reflection)
            glare_mask = np.zeros(img_size, dtype=np.float32)
            cv2.circle(glare_mask, (glare_x, glare_y), random.randint(40, 80), 
                      glare_intensity, -1)
            glare_mask = cv2.GaussianBlur(glare_mask, (51, 51), 0)
            canvas = np.clip(canvas + glare_mask, 0, 1.0)

        # 4. Add subtle sensor noise (shot noise from optical sensor)
        poisson_noise = np.random.poisson(canvas * 20).astype(np.float32) / 20
        canvas = np.clip(canvas + (poisson_noise - canvas) * 0.05, 0, 1.0)

        # 5. Convert and save
        final_img = (canvas * 255).astype(np.uint8)
        file_name = f"puf_master_{i:04d}.png"
        cv2.imwrite(os.path.join(output_dir, file_name), final_img)

    print(f"Successfully generated {num_images} PUF master images in '{output_dir}'")

# Run the generator
generate_puf_dataset(num_images=1000)