# Let's first examine the uploaded architecture diagram
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load and display the architecture diagram
img = Image.open('gen-ai-architecture.jpg')

# Display the image to understand the architecture
plt.figure(figsize=(15, 20))
plt.imshow(img)
plt.axis('off')
plt.title('HC-SCDNet Architecture Diagram', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight')
plt.show()

print("Architecture diagram loaded and displayed.")
print(f"Image dimensions: {img.size}")