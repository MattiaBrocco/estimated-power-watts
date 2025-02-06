# EstimatedPowerDataField - Garmin Simple Data Field

## Overview

This is a Garmin Simple Data Field that estimates **real-time power output (in Watt/Kg)** without requiring a physical power meter.
Starting from a physics-informed model for computing power [[1]](#1), a **Decision Tree Regressor** with custom ex-post pruning has been implemented in order to extend the first model by incorporating additional factors such as:
* Heart Rate
* Cadence
* Distance Elapsed
* Slope/Grade

The actual performance recorded on several .FIT files is the following:
| MAE |
| --- |
| 0.20 W/kg (17.40 W @ 85 Kg) |

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

## References
<a id="1">[1]</a> [Cycling power and speed](https://www.gribble.org/cycling/power_v_speed.html)

---

**Author**: [Mattia Brocco](https://www.linkedin.com/in/mattia-brocco-data-science/)<br>
**License**: MIT License<br>
**MITVersion**: v0.0<br>
<a style="display:inline-block;background-color:#FC5200;color:#fff;padding:5px 10px 5px 30px;font-size:11px;font-family:Helvetica, Arial, sans-serif;white-space:nowrap;text-decoration:none;background-repeat:no-repeat;background-position:10px center;border-radius:3px;background-image:url('https://badges.strava.com/logo-strava-echelon.png')" href='https://strava.com/athletes/105647830' target="_clean">
  Follow me on
  <img src='https://badges.strava.com/logo-strava.png' alt='Strava' style='margin-left:2px;vertical-align:text-bottom' height=13 width=51 />
</a>
