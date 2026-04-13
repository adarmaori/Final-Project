Implementing the distortion effect on the Bela platform is an excellent choice. Bela is essentially a small Linux computer (BeagleBone Black) with specialized hardware for ultra-low latency audio.

Here is the step-by-step implementation plan for your **Bela Neural Distortion**.

### 1\. The Core Concept

You will run a C++ program on the Bela board that:

1.  Reads an audio sample from the input (Analog In or Audio In).
2.  Feeds it into a lightweight Neural Network (running sample-by-sample).
3.  Writes the result to the output.

### 2\. The Library: RTNeural

Do not use TensorFlow Lite or PyTorch directly on Bela for this specific task (they are often too heavy for sample-by-sample processing).
Instead, use **RTNeural**.

  * **What it is:** A lightweight C++ library specifically designed for running neural networks in real-time audio plugins and embedded devices.
  * **Why:** It supports AVX/NEON acceleration (which Bela's ARM processor has) and uses fixed-size buffers, avoiding memory allocation during the audio callback (which causes glitches).

### 3\. Implementation Steps

#### Step A: Train the Model (on your PC)

You need to "capture" the sound of a distortion pedal.

1.  **Record Data:**
      * Send a clean guitar signal (or sine sweeps) into a distortion pedal (e.g., Tube Screamer).
      * Record the output.
      * You now have pairs of `input.wav` and `target.wav`.
2.  **Train in Python:**
      * Use PyTorch to train a simple **LSTM** or **GRU** network (try 1 layer, 8-16 units) to predict the *target* from the *input*.
      * Export the trained weights to a JSON file (RTNeural has a script for this).

#### Step B: The C++ Code (on Bela)

You will write a `render.cpp` file in the Bela IDE.

```cpp
#include <Bela.h>
#include <RTNeural/RTNeural.h> // You will need to upload this library to Bela

// Define the neural network structure
// (Example: 1 input -> GRU layer -> Dense layer -> 1 output)
RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 8>, RTNeural::DenseT<float, 8, 1>> model;

bool setup(BelaContext *context, void *userData) {
    // Load the weights file you trained in Python
    auto weights_file = std::ifstream("model_weights.json");
    model.parseJson(weights_file);
    model.reset();
    return true;
}

void render(BelaContext *context, void *userData) {
    // Process audio buffer
    for(unsigned int n = 0; n < context->audioFrames; n++) {
        // Read input (Channel 0)
        float input = audioRead(context, n, 0);
        
        // Run the Neural Network
        // (Note: For minimal latency, we process sample-by-sample)
        float output = model.forward(&input);
        
        // Write output (Channel 0 and 1)
        audioWrite(context, n, 0, output);
        audioWrite(context, n, 1, output);
    }
}
```

#### Step C: Optimization for Bela

Bela runs on an ARM Cortex-A8. To make the NN run without "clicks" (xruns):

1.  **Use NEON Instructions:** RTNeural can use the NEON SIMD unit on the BeagleBone. Make sure to enable this flag in the Makefile or compiler settings.
2.  **Keep it Small:** Start with a very small network (e.g., a standard MLP with 16 neurons). If the CPU usage (checked in the IDE) is \< 40%, you can upgrade to a GRU/LSTM for better sound.

### 4\. What to show in the Mid-Term Presentation (12.4.2026)

  * **The Artifact:** The Bela board with a guitar plugged in.
  * **The Comparison:** A physical switch (or software button) that toggles between "Clean", "Simple Math Distortion", and "Neural Distortion".
  * **The Metric:** Show the CPU load percentage on the Bela dashboard (e.g., "Our neural pedal uses only 15% CPU").

### Action Items for Now

1.  **Get the Board:** Ensure you have the Bela Starter Kit.
2.  **Clone RTNeural:** Download `jatinchowdhury18/RTNeural` from GitHub.
3.  **Run the "Hello World":** Get the `sinetone` example running on Bela to verify your audio I/O is working.

Implementing the "Simple Math Distortion" on Bela is very straightforward. This serves as your "Control Group"—the baseline performance and sound quality that your Neural Network and FPGA implementations must compete against.

We will use the **Hyperbolic Tangent ()** function. It is the industry standard for simulating the soft-clipping behavior of analog circuits.

Here is the complete implementation strategy using the **Bela GUI** so you can tweak parameters in real-time without wiring up physical potentiometers yet.

### 1. The Math

The signal flow is:


* **Input:** The guitar signal.
* **Drive:** How much we push the signal into the "ceiling" (Gain).
* **Tanh:** The function that squashes the loud peaks (creating distortion).
* **Volume:** Lowers the volume back down so it doesn't blow out your speakers.

### 2. The Code (`render.cpp`)

Copy this code into your Bela IDE project.

```cpp
#include <Bela.h>
#include <libraries/Gui/Gui.h> // Library for browser-based sliders
#include <cmath>               // Standard math library for std::tanh

// --- GUI Objects ---
Gui gui;
GuiController controller;

// --- Global Variables ---
float gDrive = 1.0f;
float gVolume = 0.5f;

bool setup(BelaContext *context, void *userData)
{
    // 1. Setup the GUI
    gui.setup(context->projectName);
    controller.setup(&gui, "Distortion Controls");

    // 2. Create Sliders in the Browser
    // Arguments: Name, Default Value, Min, Max, Increment
    controller.addSlider("Drive (Gain)", 1.0f, 1.0f, 50.0f, 0.1f);
    controller.addSlider("Output Level", 0.3f, 0.0f, 1.0f, 0.01f);

    return true;
}

void render(BelaContext *context, void *userData)
{
    // Retrieve values from the GUI sliders (Update once per block to save CPU)
    gDrive = controller.getSliderValue(0);   // Index 0 = Drive
    gVolume = controller.getSliderValue(1);  // Index 1 = Volume

    // Loop through all audio frames (samples) in the block
    for(unsigned int n = 0; n < context->audioFrames; n++)
    {
        // 1. Read Input (Channel 0 - Left)
        float input = audioRead(context, n, 0);

        // 2. Apply Drive (Gain)
        float preDistortion = input * gDrive;

        // 3. Apply The Effect (Math Distortion)
        // std::tanh creates the characteristic "soft clipping" curve
        float distorted = std::tanh(preDistortion);

        // 4. Apply Output Volume (Makeup Gain)
        float output = distorted * gVolume;

        // 5. Write to Output (Left and Right)
        audioWrite(context, n, 0, output);
        audioWrite(context, n, 1, output);
        
        // Optional: Visualize signals in the Bela Oscilloscope
        // scope.log(input, output); 
    }
}

void cleanup(BelaContext *context, void *userData)
{
    // Nothing to clean up
}

```

### 3. How to Run and Test

1. **Paste** the code into the Bela IDE.
2. **Run** the project.
3. **Open the GUI:** Click the "GUI" button in the Bela IDE toolbar (or go to `http://bela.local/gui`).
4. **Play Guitar:** Start with "Drive" low. As you increase the Drive slider, you will hear the clean signal transform into a fuzzy, saturated distortion.

### 4. Moving to Physical Potentiometers (Optional)

If you want to use real knobs instead of the browser for your demo:

1. Connect a 10k potentiometer to the Bela's **Analog In 0** (3.3V, GND, and Wiper to Input).
2. Replace `controller.getSliderValue(0)` with:
```cpp
// Map 0.0-1.0 analog input to 1.0-50.0 drive range
gDrive = 1.0f + (analogRead(context, n/2, 0) * 49.0f);

```



### 5. Why this is important for your project

When you present your project, this code allows you to make the following claim:

> "We compared our Neural Network not just against 'Clean' audio, but against the industry-standard mathematical approximation (). While  uses X% CPU, our Neural Network captures nuances of specific analog gear that a simple  curve cannot."