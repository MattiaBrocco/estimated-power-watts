# EstimatedPowerDataField - Garmin Simple Data Field

## Overview

This is a Garmin Simple Data Field that estimates **real-time power output (in Watt/Kg)** without requiring a physical power meter. The estimation is based on a **Decision Tree Regressor**, fine-tuned using a reference model called PIPPO, which defines pedal power based on core physical variables. The algorithm extends this logic by incorporating additional factors such as:

* Heart Rate
* Cadence
* Distance Elapsed
* Slope/Grade

![image](https://github.com/user-attachments/assets/cb412864-58a5-4d92-867d-5b16787b201c)

## Requirements

To use this data field, the following requirements must be met:

1. **Garmin Device**: Currently supported only on the Garmin Edge Explore 2.
2. **User Weight**: Must be set in Garmin Connect.
3. **Bike Weight**: Must be configured in the app settings within Garmin Connect IQ.
4. **Slope/Grade Data Field**: The device must have access to a slope/grade data field.

## Assumed Units of Measurement

The algorithm assumes the following units:

| Variable | Unit |
| ---------|----- |
| Altitude | meters (m) |
| Distance | meters (m) |
| Speed | meters/second (m/s) |
| Slope | percentage (%) |
| User Weight | grams (g) |

## Known Issues & Future Improvements
* **Slope Handling Bug**: The current implementation sometimes fails to read the slope/current_slope/grade data field correctly. A fix is planned for future versions.
* **Multi-Device Support**: Currently, only Edge Explore 2 is supported, but future versions may extend compatibility.
* **Language Support**: English only for now; localization may be added in future updates.

## Contribution & Feedback

Feel free to open issues or submit pull requests to improve the algorithm or add new features. Feedback is always welcome!

Author: [Mattia Brocco](https://www.linkedin.com/in/mattia-brocco-data-science/)
License: MIT License
MITVersion: v0.0
