# Nvidia Jetson Orin Nano jetracer
I created an autonomous AI race car by attaching the NVIDIA Jetson Orin Nano, which happens to be one of the most powerful edge AI computers developed by NVIDIA, to the chassis of a 1/18 scale LaTrax Rally RC car by fabricating my own plastic mounting plate to attach all necessary electronic components, including the computer board, PCA9685 servo driver, and signal multiplexer, without compromising the integrity of the chassis. In terms of hardware development, the greatest problem encountered was with drivetrain development when a cold solder joint, as well as overapplication of heat-shrink tubing to the bullet connectors between the ESC and the motor, was preventing the transfer of electricity due to load, as determined through a careful process of testing with pin swapping on each PWM. The greatest lesson learned from developing a project on such complex systems level is that true engineering is not about programming your code from scratch but about understanding the architecture of an existing system and determining how to fix it on nonstandard hardware.



| **Engineer** | **School** | **Area of Interest** | **Grade** | 
|:--:|:--:|:--:|:--:|
| Gautham N.K | Bellarmine College Prepratory | Mechanical Engineering | Incoming Junior

# Milestone 3: Autonomous Navigation & Neural Network Integration

## Project Overview

For Milestone 3, the project transitions into full autonomous navigation using an onboard camera, an Nvidia Jetson compute platform, and a Convolutional Neural Network (CNN). The vehicle operates in two distinct modes:

1. **Autonomous Mode:** The front-facing camera acts as the visual sensor, continuously streaming track frames into the Nvidia Jetson ("the brain"). The Jetson runs a trained CNN regression model to process image data in real time, determining spatial trajectories and outputting steering targets.
2. **Manual Driver Mode:** A 2.4 GHz radio transmitter communicates with an onboard receiver. A hardware multiplexer routes throttle and steering commands dynamically — directing steering signals to the servo motor and throttle signals to the Electronic Speed Controller (ESC).

---

## Convolutional Neural Network (CNN) Deep Dive

### 1. Why Regression Instead of Classification

Most introductory CNN projects are classifiers — "is this a stop sign or a light?" — where the output is a discrete label pulled from a fixed set of categories via a softmax layer. Our navigation problem doesn't fit that mold: there's no finite set of "correct" track positions, only a continuous 2D space of possible target points on the image plane. So instead of a softmax over categories, the final layer is a **linear regression head** that outputs continuous (x, y) coordinates — normalized to a -1 to 1 range so the model's output is independent of the camera's native resolution. This is the same architectural family used in NVIDIA's JetRacer/JetBot road-following examples, and it's what lets a single forward pass through the network double as a steering command generator rather than a label picker.

### 2. Input Representation

Each camera frame is ingested as a 3D tensor: **Width × Height × Color Channels** (typically resized to 224×224×3 to match the input dimensions expected by pretrained ImageNet backbones like ResNet18). Before entering the network, each frame is normalized per-channel (mean/std matching the pretrained backbone's original training distribution) so that pixel intensity fluctuations from track lighting don't dominate over actual track geometry.

### 3. Feature Extraction — Convolutional Layers

The convolutional layers are the model's "eyes." Each layer slides a small matrix filter (kernel) — commonly 3×3 or 7×7 — across the image, computing a dot product at every spatial position. Early layers in the network tend to pick up low-level primitives: edges, color boundary lines, and contrast gradients (exactly the kind of thing that distinguishes track surface from track margin). As frames pass through deeper layers, the kernels start responding to more abstract, composite patterns — curvature of the track boundary, the vanishing point of a straightaway, or the visual "wedge" shape formed by two converging lane edges. Each convolutional layer also increases the number of channels (feature maps) while the spatial resolution shrinks, trading raw pixel detail for a richer description of *what* is in a given region.

### 4. Non-Linearity — ReLU

Without a non-linear activation function, stacking convolutional layers would collapse mathematically into a single linear operation, no matter how many layers you add — the network would only be capable of learning simple linear relationships between input pixels and output coordinates. The Rectified Linear Unit, `f(x) = max(0, x)`, is inserted after each convolution specifically to break that linearity. It's computationally cheap (just a threshold at zero) and it's what allows the network to model the actual non-linear geometry of a curving track — a relationship that a purely linear model could never approximate no matter how much training data it saw.

### 5. Pooling — Downsampling for Spatial Invariance

Pooling layers (typically max-pooling) reduce the spatial dimensions of each feature map, usually by taking the maximum value within a small window (e.g., 2×2) and discarding the rest. This does two things for us:

- **Reduces computational load**, which matters directly for real-time inference speed on the Jetson — fewer pixels to process per layer means a faster forward pass and lower steering latency.
- **Builds tolerance to small positional shifts.** Because the chassis vibrates and bounces slightly as it drives, the exact pixel location of a track edge shifts frame to frame even when the vehicle's actual position hasn't meaningfully changed. Pooling makes the network's output stable against these small physical jitters instead of over-reacting to them.

### 6. Dense / Output Layer

After the convolutional and pooling stages, the resulting feature maps are flattened into a single vector and passed through one or more fully connected (dense) layers, terminating in a final linear layer with exactly 2 outputs (or `2 × number of target categories`, if predicting multiple points). Those two numbers are interpreted directly as the (x, y) coordinates of the target point, rendered visually during testing as a target dot overlaid on the live camera feed. The steering controller then computes the horizontal delta between image center and that target x-coordinate, scales it by a steering gain, and issues the corresponding servo command — closing the loop between "what the camera sees" and "how the wheels turn."

### 7. Training Pipeline

- **Loss function:** Mean Squared Error (MSE) between the predicted (x, y) and the human-labeled ground truth point for each training frame. MSE is a natural fit for regression because it penalizes large deviations more heavily than small ones, pushing the model toward tighter tracking rather than "roughly close" predictions.
- **Optimizer:** Adam, chosen for its adaptive per-parameter learning rate, which tends to converge faster than plain SGD on small, custom datasets like ours.
- **Transfer learning:** Rather than training a CNN from scratch, we start from a ResNet18 backbone pretrained on ImageNet and replace only its final fully connected layer with our 2-output regression head. This lets the network reuse millions of images' worth of general-purpose edge/shape/texture knowledge, and only needs to learn the track-specific mapping — a major reason a dataset of only ~150 images was enough to get a usable initial model.
- **Data augmentation:** Random horizontal flips (with the corresponding x-label negated) and color jitter (brightness/contrast/saturation) were applied during training to artificially expand the effective dataset size and make the model more robust to lighting changes across different runs on the same track.
- **Iterative expansion:** Initial training runs (~150 images) produced steering predictions that settled into a rough but noisy target vector, with visible tracking drift in areas the dataset under-represented (sharp corners, shadowed sections). Additional targeted data collection in those specific failure regions — rather than blindly adding more random frames — was the more effective lever for reducing drift, since it directly patched the gaps the model was weakest on.

### 8. Real-Time Inference Constraints

Unlike a lot of CNN applications where inference can run offline, our model has to produce a new steering target on every incoming frame while the vehicle is moving — meaning the whole pipeline (frame capture → preprocessing → forward pass → steering computation → servo command) has to complete well within the camera's frame interval. This is part of why the Jetson platform (with onboard GPU acceleration) and a comparatively lightweight backbone like ResNet18 were chosen over deeper, more accurate but slower architectures — on this project, inference latency directly translates into how sharply the car can react to sudden track curvature.

---

## Hardware Modifications & Iterations

- **Battery & Power Management:** Remounted peripheral hardware to make room for a dedicated 3S LiPo battery, ensuring consistent voltage supply to the Nvidia Jetson during heavy computational loads.
- **Traction & Mechanical Tuning:** Upgraded to high-contact foam tires. Preventing mechanical wheel slip eliminates noisy displacement data, ensuring consistent correlation between visual frame inputs and vehicle position.

---

## New Addition: Active Rear Wing with IMU-Based Stabilization

### 1. Motivation

At higher test speeds, the chassis experiences small pitch and roll oscillations from track imperfections, cornering load transfer, and motor vibration — all of which subtly change how much the camera's field of view "bounces" frame to frame, and can also unsettle rear traction going into corners. To address this, we're adding a **servo-actuated active rear wing** that adjusts its angle of attack in real time based on the vehicle's measured dynamics, rather than sitting at a fixed angle like a static wing. The goal is added rear-end stability under braking and cornering load, using the same kind of active-aero concept found in full-scale race cars, scaled down to our platform.

### 2. Sensing: Accelerometer / IMU Placement

A small IMU (accelerometer + gyroscope) is mounted near the vehicle's center of mass, oriented to capture:

- **Longitudinal acceleration** — braking and throttle events, which the wing can respond to by increasing angle under hard braking (added rear downforce/drag) and flattening under acceleration (reduced drag).
- **Lateral acceleration** — cornering load, letting the wing add a modest angle increase through hard corners for rear grip.

### 3. The Core Problem: Raw IMU Data Is Noisy

Raw accelerometer output is inherently jittery — vibration from the motor, chassis flex, and the foam tires' own compliance all inject high-frequency noise into the signal that has nothing to do with the vehicle's actual dynamic state. If the wing's servo were commanded directly off raw accelerometer readings, it would chatter constantly — reacting to noise spikes rather than genuine acceleration events — leading to premature servo wear and an unstable, twitchy wing that could do more harm than good aerodynamically.

### 4. Noise Reduction Strategy

Two complementary techniques address this:

**a) Low-pass filtering.** Before any control decision is made, raw accelerometer samples pass through a low-pass filter (a simple moving average or an exponential/complementary filter blended with the gyroscope), attenuating the high-frequency vibration noise while preserving the slower, real dynamic trends we actually care about (a genuine braking or cornering event happens over hundreds of milliseconds, not the sub-millisecond timescale of vibration noise).

**b) Hysteresis on the control decision.** Filtering alone doesn't fully solve the problem — a signal that's sitting right near a threshold can still flicker back and forth across it, causing the servo to twitch between two wing angles. Hysteresis solves this by using **two separate thresholds instead of one**: the wing only moves to a "deployed" angle once acceleration exceeds an upper threshold, and only returns to neutral once it drops below a distinctly lower threshold. That dead-band gap between the two thresholds means a signal hovering near a single trigger point can't cause rapid back-and-forth switching — it has to clearly cross into the new state and clearly leave the old one before the wing responds again. In practice this looks like:

# Milestone 2: Hardware Validation & PWM Calibration
 <iframe width="985" height="554" src="https://www.youtube.com/embed/oSRA-IN0WpY" title="Gautham N. K. Milestone 2" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


 ## Calibration and Hardware Verification

This step is the essential link between the construction of the physical parts of the car and the software which controls it in the autonomous vehicle system architecture. With a combination of the Jetson Orin Nano along with the PCA9685 PWM Driver and Pololu Servo Multiplexer, the objective was to set up successful communication between the RC receiver, the electronics, steering servo and ESC. Calibration makes sure that the PWM signals from the software translate into the right hardware output before computer vision.

## Diagnosing the Throttle Anomaly 

As part of the early hardware testing phase, an important problem was identified: steering signals operated the servo properly but the motor did not respond to throttle signals even though it seemed like the ESC was receiving power. In order to identify the exact cause of the problem, a pin-swapping troubleshooting technique was used on the PCA9685 PWM channels – exchanging the pins of the steering and throttle signal connections proved that the PWM signals were working properly on all channels, ruling out any problems with the control board and remote control. The problem was then traced down to the actual mechanical drive. As a result of a thorough analysis of the wiring, two interrelated faults in the connection between the ESC and the motor were discovered: a bad cold solder joint where there was not enough tin flow resulting in a poor conductivity and heat shrink tubing wrapped too tightly around the connector that physically compressed the contacts and limited the current flow. While during bench testing the connection was fine, under the motor load the voltage drop across the connection was enough to prevent the ESC from functioning.

## PWM Calibration via Jupyter

Since the physical configuration of the robot has been successfully verified, the robot underwent calibration using the Jupyter Notebooks interacting with the Orin Nano in Wi-Fi mode. The precise values of pulse widths for the actuator channels have been determined. The steering calibration was performed by determining the optimal deflection angle and the center point offset in order to make sure that the minimum possible radius of curvature could be achieved for tight indoor tracks while avoiding any jamming of the wheels in full lock. Throttle calibration determined the following thresholds: the neutral deadband during which the ESC will stay idle without moving, the minimal pulse width needed for overcoming drivetrain friction, and the maximal acceleration without causing the wheels to slip on the indoor surface.

# Milestone 1: Assembly of the Hardware – Creating the JetRacer Platform
<iframe width="1285" height="723" src="https://www.youtube.com/embed/wCwTizB-BSY" title="Gautham N. K. Milestone 1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Summary

In this milestone, I managed to assemble the whole physical platform of a completely autonomous JetRacer which is an RC car platform reconfigured to become an edge AI prototyping platform that utilizes an NVIDIA Jetson Nano board. The final product will be able to perceive the track via a camera and make decisions on how to control steering and acceleration based on a computer vision model running on this platform itself but will remain under my manual control at any point I desire.

## What I Created

To create a JetRacer platform, I took apart an ordinary RC car chassis in order to find out the main electronic elements:


The ESC (Electronic Speed Controller) – the part that gets an input signal and turns it into a command for the motor. In other words, it is a "throttle translator" between electronics and a motor.
The radio receiver – the electronic board that gets commands from an RC remote controlled by a human.

The Radio Receiver – the part of the system that receives data from a radio remote controlled by a person, enabling a person to control the car manually via the remote over radio waves, similar to an ordinary RC car.


## Next I performed the following steps:


Added the multiplexer (Mux) into the data line. The Mux acts as a switch that selects one out of multiple signals and routes it to the output. In this case, the Mux is between the RC receiver, the Jetson Nano and the ESC/servo, allowing to switch a switch that will control whether the AI or a human via the remote is in charge. This is the fundamental safety element of the entire project – when the autonomous system fails in any way, it can be overridden instantly by the manual system.
Connected the servo lines. The servo motor is responsible for controlling the steering angle. Connecting it to the Mux was required to enable both the AI and the manual remote control to control the steering angle accordingly.
This hardware setup forms the fundamental data control loop of the entire project:

Camera detects the line → Jetson Nano processes the line and makes steering/ throttle decisions → mux sends those decisions to the servo and ESC → car drives with the human being able to take control of anything by using the mux.

## The Technical Challenge: QSPI Firmware

The toughest aspect of this phase didn't lie in the wiring but in making sure the Jetson Nano could boot up properly in the first place.

The device had been idle for more than a year and I had thought that inserting an entirely new MicroSD card with the proper OS image would have been sufficient to get it started, just like with other devices. But it was not the case, the Jetson Nano couldn't start with a newly flashed MicroSD card.

The problem is simply that of one distinction which people do not usually make when dealing with ordinary computers; MicroSD card contains the operating system but not the firmware which instructs the board how to boot in the first place. The startup firmware in question is stored on another physical memory chip on the board, QSPI flash (Quad Serial Peripheral Interface – a small, quick-access memory chip which is soldered on the board). In other words, the difference may be illustrated by the case when a car will not start if its ignition system (QSPI firmware) is out of date even though there is new fuel in the tank (new SD card/OS).

In the case when my board stayed unoperated for quite some time, the QSPI firmware on it became out of date and thus not compatible with the OS image which I attempted to load. I needed to:

Figure out that the problem is indeed a QSPI firmware problem, not the SD card nor some problem with the wiring itself.
Find a new version of QSPI firmware that can be uploaded directly to the chip using NVIDIA's recovery mode.
Boot from the new QSPI firmware successfully before the operating system from the SD card can load at all.


The updated QSPI firmware resolved the booting problems, and now the system runs perfectly fine.

## Next Steps

Having got my hardware assembled and my Jetson Nano working, the next steps are to achieve:


Setting up hardware communication — establishing the software-level communication (like I2C, UART, or GPIO protocols) allowing the Jetson Nano, the multiplexer, the servo, and the ESC to communicate with one another.
Testing computer vision models — testing different computer vision models to find out which works best to interpret the camera stream and steer the car through the track, taking into account both precision and speed.


This milestone is the foundation for the future work — nothing further can be done without the properly working hardware and the safety-first control switch.

# Hardware flowcharts

## Wiring Diagram 
![Wiring Diagram](CustomWiringDiagram.jpg)

## Power Architecture Flow Chart
![Flow Chart](PowerArchitecture.png
)

## Signal flow Flow Chart
![Flow Chart](SignalFlow.png)

# Different applicable notebooks for training

## Interactive regression Notebook

## Computer Vision & Interactive Regression Code

Below is the script for interactive regression dataset collection, model setup (ResNet18 backbone), live inference loop, and training interface:

```python
# ==============================================================================
# Interactive Regression Notebook (interactive_regression.py)
# Source: NVIDIA-AI-IOT/jetracer
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. CAMERA INITIALIZATION
# ------------------------------------------------------------------------------
from jetcam.csi_camera import CSICamera
# from jetcam.usb_camera import USBCamera

camera = CSICamera(width=224, height=224)
camera.running = True


# ------------------------------------------------------------------------------
# 2. TASK & DATASET CONFIGURATION
# ------------------------------------------------------------------------------
import torchvision.transforms as transforms
from xy_dataset import XYDataset

TASK = 'road_following'
CATEGORIES = ['apex']
DATASETS = ['A', 'B']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets = {}
for name in DATASETS:
    datasets[name] = XYDataset(TASK + '_' + name, CATEGORIES, TRANSFORMS, random_hflip=True)


# ------------------------------------------------------------------------------
# 3. DATA COLLECTION WIDGET
# ------------------------------------------------------------------------------
import cv2
import ipywidgets
import traitlets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
from jupyter_clickable_image_widget import ClickableImageWidget

dataset = datasets[DATASETS[0]]
camera.unobserve_all()

camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)
snapshot_widget = ipywidgets.Image(width=camera.width, height=camera.height)
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

dataset_widget = ipywidgets.Dropdown(options=DATASETS, description='dataset')
category_widget = ipywidgets.Dropdown(options=dataset.categories, description='category')
count_widget = ipywidgets.IntText(description='count')

count_widget.value = dataset.get_count(category_widget.value)

def set_dataset(change):
    global dataset
    dataset = datasets[change['new']]
    count_widget.value = dataset.get_count(category_widget.value)

dataset_widget.observe(set_dataset, names='value')

def update_counts(change):
    count_widget.value = dataset.get_count(change['new'])

category_widget.observe(update_counts, names='value')

def save_snapshot(_, content, msg):
    if content['event'] == 'click':
        data = content['eventData']
        x = data['offsetX']
        y = data['offsetY']
        dataset.save_entry(category_widget.value, camera.value, x, y)
        snapshot = camera.value.copy()
        snapshot = cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3)
        snapshot_widget.value = bgr8_to_jpeg(snapshot)
        count_widget.value = dataset.get_count(category_widget.value)

camera_widget.on_msg(save_snapshot)

data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget, snapshot_widget]),
    dataset_widget,
    category_widget,
    count_widget
])

display(data_collection_widget)


# ------------------------------------------------------------------------------
# 4. MODEL DEFINITION & LOAD/SAVE CONTROLS
# ------------------------------------------------------------------------------
import torch
import torchvision

device = torch.device('cuda')
output_dim = 2 * len(dataset.categories)

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, output_dim)
model = model.to(device)

model_save_button = ipywidgets.Button(description='save model')
model_load_button = ipywidgets.Button(description='load model')
model_path_widget = ipywidgets.Text(description='model path', value='road_following_model.pth')

def load_model(c):
    model.load_state_dict(torch.load(model_path_widget.value))

model_load_button.on_click(load_model)

def save_model(c):
    torch.save(model.state_dict(), model_path_widget.value)

model_save_button.on_click(save_model)

model_widget = ipywidgets.VBox([
    model_path_widget,
    ipywidgets.HBox([model_load_button, model_save_button])
])

display(model_widget)


# ------------------------------------------------------------------------------
# 5. LIVE EXECUTION THREAD
# ------------------------------------------------------------------------------
import threading
import time
from utils import preprocess
import torch.nn.functional as F

state_widget = ipywidgets.ToggleButtons(options=['stop', 'live'], description='state', value='stop')
prediction_widget = ipywidgets.Image(format='jpeg', width=camera.width, height=camera.height)

def live(state_widget, model, camera, prediction_widget):
    global dataset
    while state_widget.value == 'live':
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed).detach().cpu().numpy().flatten()
        category_index = dataset.categories.index(category_widget.value)
        x = output[2 * category_index]
        y = output[2 * category_index + 1]
        
        x = int(camera.width * (x / 2.0 + 0.5))
        y = int(camera.height * (y / 2.0 + 0.5))
        
        prediction = image.copy()
        prediction = cv2.circle(prediction, (x, y), 8, (255, 0, 0), 3)
        prediction_widget.value = bgr8_to_jpeg(prediction)

def start_live(change):
    if change['new'] == 'live':
        execute_thread = threading.Thread(target=live, args=(state_widget, model, camera, prediction_widget))
        execute_thread.start()

state_widget.observe(start_live, names='value')

live_execution_widget = ipywidgets.VBox([
    prediction_widget,
    state_widget
])

display(live_execution_widget)


# ------------------------------------------------------------------------------
# 6. TRAINING & EVALUATION PIPELINE
# ------------------------------------------------------------------------------
BATCH_SIZE = 8
optimizer = torch.optim.Adam(model.parameters())

epochs_widget = ipywidgets.IntText(description='epochs', value=1)
eval_button = ipywidgets.Button(description='evaluate')
train_button = ipywidgets.Button(description='train')
loss_widget = ipywidgets.FloatText(description='loss')
progress_widget = ipywidgets.FloatProgress(min=0.0, max=1.0, description='progress')

def train_eval(is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, eval_button, train_button, accuracy_widget, loss_widget, progress_widget, state_widget
    try:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        state_widget.value = 'stop'
        train_button.disabled = True
        eval_button.disabled = True
        time.sleep(1)
        
        if is_training:
            model = model.train()
        else:
            model = model.eval()
            
        while epochs_widget.value > 0:
            i = 0
            sum_loss = 0.0
            for images, category_idx, xy in iter(train_loader):
                images = images.to(device)
                xy = xy.to(device)
                
                if is_training:
                    optimizer.zero_grad()
                    
                outputs = model(images)
                
                loss = 0.0
                for batch_idx, cat_idx in enumerate(list(category_idx.flatten())):
                    loss += torch.mean((outputs[batch_idx][2 * cat_idx:2 * cat_idx+2] - xy[batch_idx])**2)
                loss /= len(category_idx)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
                    
                count = len(category_idx.flatten())
                i += count
                sum_loss += float(loss)
                progress_widget.value = i / len(dataset)
                loss_widget.value = sum_loss / i
                
            if is_training:
                epochs_widget.value = epochs_widget.value - 1
            else:
                break
    except Exception as e:
        pass
        
    model = model.eval()




