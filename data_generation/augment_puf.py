import numpy as np
import cv2
import os
import glob
import random


def estimate_tape_background(img, ksize=31):
    """
    Estimate local tape background by suppressing small bright particle dots.
    """
    ksize = max(3, int(ksize) | 1)
    return cv2.medianBlur(img, ksize)

def apply_perspective_warp(img, severity=1.0):
    """
    Simulates a phone being held at a slight angle during scanning.
    Low severity preserves the particle pattern well - represents authentic scanning variation.
    """
    h, w = img.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Slight perspective changes from hand-held scanning angles
    m = int(15 * severity)  # Max corner shift in pixels
    pts2 = np.float32([
        [random.randint(-m, m), random.randint(-m, m)],
        [w - random.randint(-m, m), random.randint(-m, m)],
        [random.randint(-m, m), h - random.randint(-m, m)],
        [w - random.randint(-m, m), h - random.randint(-m, m)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    border_val = int(np.median(estimate_tape_background(img, ksize=31)))
    return cv2.warpPerspective(img, matrix, (w, h), borderValue=border_val)

def apply_wear_degradation(img):
    """
    Simulates normal wear and tear: heat from storage/shipping, natural fading over time.
    Particles fade slightly, subtle contrast loss.
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Thermal degradation: brighten the tape, fade the particles
    # Simulate adhesive darkening and particle visibility loss
    fade_factor = np.random.uniform(0.92, 0.98)
    img_float = img_float * fade_factor
    
    # Add subtle colored cast (adhesive oxidation)
    tape_color_shift = np.random.uniform(-0.02, 0.01)
    img_float = np.clip(img_float + tape_color_shift, 0, 1)
    
    # Subtle thermal blur (adhesive flow over time) - very light
    if random.random() > 0.5:
        img_float = cv2.GaussianBlur(img_float, (3, 3), 0.3)
    
    return (img_float * 255).astype(np.uint8)

def apply_thermal_tampering(img):
    """
    Simulates heating to loosen the adhesive layer, then peeling up and replacing the tape.
    Results in displaced/realigned particles and possible particle loss.
    """
    h, w = img.shape
    
    # Create a mapping grid for local distortion
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    # Define a region that was peeled (often multiple regions from thermal stress)
    tamper_zones = random.randint(1, 3)
    for _ in range(tamper_zones):
        # Random location of peel
        tx = random.randint(w // 6, 5 * w // 6)
        ty = random.randint(h // 6, 5 * h // 6)
        radius = random.randint(50, 120)
        
        # Create distortion field: particles shift as glue re-adheres
        # Non-uniform shift simulates the peel-replace action
        dist = np.sqrt((map_x - tx)**2 + (map_y - ty)**2)
        mask = dist < radius
        
        # Radial displacement: stronger at center, weakens at edges
        strength = (1 - dist[mask] / radius) ** 1.5
        angle = np.arctan2(map_y[mask] - ty, map_x[mask] - tx)
        
        displacement = np.random.uniform(8, 20) * strength
        map_x[mask] += np.cos(angle) * displacement
        map_y[mask] += np.sin(angle) * displacement
    
    # Apply the remap (non-linear warping)
    border_val = int(np.median(estimate_tape_background(img, ksize=31)))
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=border_val)
    
    # Particle loss in tamper regions: some particles fall off during peel.
    # Fill with local background tone (not black) to avoid unrealistic square artifacts.
    bg = estimate_tape_background(warped, ksize=31)
    for _ in range(random.randint(2, 5)):
        loss_x = random.randint(0, w - 40)
        loss_y = random.randint(0, h - 40)
        size = random.randint(25, 50)

        x1, y1 = loss_x, loss_y
        x2, y2 = min(w, loss_x + size), min(h, loss_y + size)
        if x2 <= x1 or y2 <= y1:
            continue

        region = warped[y1:y2, x1:x2].astype(np.float32)
        bg_region = bg[y1:y2, x1:x2].astype(np.float32)
        rh, rw = region.shape

        # Soft-edged elliptical mask so removal blends naturally.
        mask = np.zeros((rh, rw), dtype=np.float32)
        center = (rw // 2, rh // 2)
        axes = (max(3, rw // 3), max(3, rh // 3))
        cv2.ellipse(mask, center, axes, random.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)

        blended = region * (1.0 - mask) + bg_region * mask
        warped[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return warped

def apply_physical_cut(img):
    """
    Simulates cutting the tape with scissors or a knife.
    Creates sharp discontinuities with particle damage at the cut edge.
    """
    h, w = img.shape
    img_result = img.copy().astype(np.float32)
    bg = estimate_tape_background(img, ksize=31).astype(np.float32)
    
    # Randomly choose cut orientation: horizontal, vertical, or diagonal
    cut_type = random.choice(['horizontal', 'vertical', 'diagonal'])
    
    if cut_type == 'horizontal':
        cut_y = random.randint(h // 4, 3 * h // 4)
        # Cut line with slight waver (not perfectly straight)
        x_offsets = np.sin(np.arange(w) / 10) * random.uniform(1, 3)
        for x in range(w):
            y = int(cut_y + x_offsets[x])
            if 0 <= y < h:
                # Remove particles in a thin band at the cut using background-matched fill.
                band = random.randint(3, 6)
                y1 = max(0, y - band)
                y2 = min(h, y + band)
                img_result[y1:y2, x] = bg[y1:y2, x]
    
    elif cut_type == 'vertical':
        cut_x = random.randint(w // 4, 3 * w // 4)
        y_offsets = np.sin(np.arange(h) / 10) * random.uniform(1, 3)
        for y in range(h):
            x = int(cut_x + y_offsets[y])
            if 0 <= x < w:
                band = random.randint(3, 6)
                x1 = max(0, x - band)
                x2 = min(w, x + band)
                img_result[y, x1:x2] = bg[y, x1:x2]
    
    else:  # diagonal
        # Diagonal cut from corner to opposite corner
        start_x = random.randint(0, w // 3)
        end_x = random.randint(2 * w // 3, w)
        start_y = random.randint(0, h // 3)
        end_y = random.randint(2 * h // 3, h)
        
        for t in np.linspace(0, 1, max(h, w)):
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            if 0 <= x < w and 0 <= y < h:
                band = random.randint(3, 6)
                y1 = max(0, y - band)
                y2 = min(h, y + band)
                x1 = max(0, x - band)
                x2 = min(w, x + band)
                img_result[y1:y2, x1:x2] = bg[y1:y2, x1:x2]
    
    return np.clip(img_result, 0, 255).astype(np.uint8)

def apply_scanning_noise(img):
    """
    Adds realistic scanning artifacts: minor optical sensor noise, slight compression artifacts.
    These should be subtle and preserve particle pattern integrity.
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Sensor Gaussian noise (very low)
    gaussian_noise = np.random.normal(0, 1.5, img.shape)
    
    # Poisson noise (shot noise from light sensor)
    poisson_noise = np.random.poisson(img_float * 30).astype(np.float32) / 30 - img_float
    
    result = img_float + gaussian_noise/255.0 + poisson_noise * 0.03
    return (np.clip(result, 0, 1.0) * 255).astype(np.uint8)

def process_dataset(input_dir="dataset", output_dir="paired_dataset"):
    """
    Generate synthetic paired dataset with authentic positives and diverse negatives.
    
    - Positive: Same tape scanned again (slight perspective, normal wear, scanning noise)
    - Negative: Different tampering scenarios (cut, thermal warp, heat degradation)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    master_files = glob.glob(os.path.join(input_dir, "puf_master_*.png"))
    total_generated = 0
    
    for idx, fpath in enumerate(master_files):
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {fpath}")
            continue
            
        base_name = os.path.basename(fpath).replace(".png", "")
        
        # === POSITIVE SAMPLES ===
        # These represent the same tape scanned again: should match the master
        # Includes natural wear from storage/shipping in hot environments
        num_positives = random.randint(4, 6)
        for p_idx in range(num_positives):
            pos = apply_perspective_warp(img, severity=0.7)  # Slight angle variation
            pos = apply_wear_degradation(pos)  # Minimal thermal fade
            pos = apply_scanning_noise(pos)  # Add realistic sensor noise
            
            cv2.imwrite(
                os.path.join(output_dir, f"{base_name}_positive_{p_idx:02d}.png"), 
                pos
            )
            total_generated += 1
        
        # Heat degradation from natural storage (also positive - authentic tape aging)
        num_heat_natural = random.randint(1, 2)
        for h_idx in range(num_heat_natural):
            pos = img.copy().astype(np.float32) / 255.0
            
            # Moderate fading from storage heat (not aggressive)
            fade_factor = np.random.uniform(0.85, 0.95)
            pos = pos * fade_factor
            
            # Light thermal blur from adhesive aging
            for _ in range(random.randint(1, 2)):
                pos = cv2.GaussianBlur(pos, (3, 3), 0.4)
            
            # Subtle thermal discoloration
            pos += np.random.uniform(-0.02, 0.01)
            pos = apply_scanning_noise((np.clip(pos, 0, 1.0) * 255).astype(np.uint8))
            
            cv2.imwrite(
                os.path.join(output_dir, f"{base_name}_positive_heat_{h_idx:02d}.png"), 
                pos
            )
            total_generated += 1
        
        # === NEGATIVE SAMPLES ===
        # These represent tampering attempts: should NOT match
        
        # 1. Thermal tampering (peel and re-stick with particle displacement)
        num_thermal = random.randint(2, 3)
        for t_idx in range(num_thermal):
            neg = apply_perspective_warp(img, severity=0.5)
            neg = apply_thermal_tampering(neg)
            neg = apply_scanning_noise(neg)
            
            cv2.imwrite(
                os.path.join(output_dir, f"{base_name}_thermal_tamper_{t_idx:02d}.png"), 
                neg
            )
            total_generated += 1
        
        # 2. Physical cut (with sharp discontinuities and particle damage)
        num_cuts = random.randint(2, 4)
        for c_idx in range(num_cuts):
            neg = apply_physical_cut(img)
            neg = apply_scanning_noise(neg)
            
            cv2.imwrite(
                os.path.join(output_dir, f"{base_name}_cut_{c_idx:02d}.png"), 
                neg
            )
            total_generated += 1

    print(f"Processed {len(master_files)} master PUFs")
    print(f"Generated {total_generated} synthetic training samples in '{output_dir}'")

process_dataset()