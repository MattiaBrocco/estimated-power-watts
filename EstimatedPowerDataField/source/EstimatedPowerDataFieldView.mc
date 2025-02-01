using Toybox.Lang;
using Toybox.Time;
using Toybox.System;
using Toybox.WatchUi;
using Toybox.AntPlus;
using Toybox.Activity;
using Toybox.UserProfile;
using Toybox.Math as Math;
using Toybox.FitContributor;

// ASSUMPTIONS:
// altitude: m
// distance: m
// speed: m/s
// slope: %
// UserWeight: g


// define all constants used in the Data Field
const POWER_UNITS = "W"; // sure?
const POWER_RECORD_ID = 0; // sure?
const POWER_NATIVE_NUM_RECORD_MESG = 0; // sure?

const userWeight = UserProfile.getProfile().weight / 1000;

// constant from physics
// Aerodynamic drag
const rho = 1.2; // air density
const C_d = 0.8; // drag coefficient
const A = 0.4; // frontal area

// Rolling resistance
const C_rr = 0.00375; // example value for coefficient of rolling resistance
const g = 9.81;
const eta = 0.95; // drivetrain efficiency

// Constant for model prediction
const THRESHOLDS = [43.5, 25.5, 16.5, -2.0, 4.334000110626221, 99.1243667602539, 245.4000015258789, 136.70000457763672, -2.0, -2.0, -2.0, -2.0, -2.0, 6.050999879837036, 43.67743492126465, 34.07872009277344, -2.0, -2.0, 89.90661239624023, 35.5, -2.0, 0.3795614689588547, -2.0, -2.0, 234.0999984741211, -2.0, -2.0, -2.0, 159.5, 137.5, 42.35653114318848, -1.592462182044983, 104.39999771118164, 31.70259475708008, 11.463000297546388, -2.0, -2.0, 81.0999984741211, -2.0, -2.0, -2.0, 89.5, 129.5, 7.608999967575073, 123.5, 50.5, 89.0999984741211, -2.0, -2.0, 38.63678550720215, -2.0, -2.0, -2.0, 85.5, 60.5, -2.0, -2.0, -2.0, 82.5, 96.0999984741211, 53.5, -2.0, 1.9462173581123352, 4.226999998092651,
                    -2.0, -2.0, -2.0, -2.0, 24.91255474090576, 24.15157985687256, 0.4153399914503097, -2.0, -2.0, -2.0, 87.70000076293945, -2.0, -2.0, 87.39999771118164, 125.5, -2.0, 8.468000411987305, 1.0927284359931946, -2.0, -2.0, -2.0, 9.456500053405762, 1.1578096747398376, -2.0, -2.0, -2.0, 84.5, 51.56657028198242, -1.0022787153720856, 45.49306488037109, -2.0, -2.0, 0.0521852746605873, -2.0, 51.41390419006348, 3.601348042488098, 42.75065612792969, -2.0, 1.7695695757865906, 5.733500003814697, -2.0, -2.0, 46.6965446472168, -2.0, -2.0, -2.0, 2.06548273563385, -2.0, -2.0, 3.9049999713897705, 58.49517440795898, 2.2888428270816803, 47.5, -2.0, -2.0, 53.57981491088867, -2.0, -2.0,
                    100.14324951171876, -2.0, -2.0, -4.0915586948394775, 71.57314872741699, -4.334704637527466, -8.045689344406128, -2.0, -2.0, -2.0, 8.82200002670288, -2.0, -2.0, 109.26069259643556, 77.5, 85.54010772705078, 272.3000030517578, 67.5, 5.607500076293945, -2.0, -1.477114200592041, -2.0, -0.620302826166153, -2.0, 74.25558471679688, -2.0, -2.0, 4.17549467086792, 10.571500301361084, 128.5, 9.843999862670898, -2.0, 126.5, -2.0, -2.0, -2.0, -2.5957677364349365, -2.0, -2.0, 69.5, -2.0, -2.0, -2.0, -3.45429003238678, -2.0, 93.5806884765625, 122.89999771118164, 120.9000015258789, -2.0, -2.0, -2.0, 7.902999877929687, 104.23472213745116, 5.477499961853027, -2.0, -2.0, -2.0, -2.0,
                    70.10932159423828, 68.10277938842773, 103.79999923706056, -2.0, 271.1000061035156, 11.649499893188477, 205.29999542236328, 201.8000030517578, -2.0, -2.0, 58.703046798706055, -2.0, -2.0, -2.0, -2.0, -2.0, 10.239999771118164, -2.0, -2.0, 63.0, 5.141499996185303, -2.0, -2.0, -2.0, 45.927024841308594, -2.0, 92.5, 2.287210822105408, 135.5, 74.16850280761719, 101.5, 71.72296524047852, -2.0, -2.0, 111.5, 109.20000076293944, 12.474999904632568, -2.0, -2.0, -2.0, -2.0, 76.74779510498047, -2.0, -2.0, 1.588510036468506, -2.0, -2.0, -2.0, 131.5, -2.0, -2.0, 145.5, 82.5, 67.5, 2.316513538360596, 2.365000009536743, -2.0, 0.9105483293533324, 58.50135040283203, -2.0, -2.0, 131.5,
                    -2.0, -2.0, 52.26077461242676, 6.032446622848511, -2.0, -2.0, 8.285499811172485, 4.278152227401733, 72.16261291503906, -2.0, -2.0, 4.705365180969238, -2.0, -2.0, -2.0, 171.70000457763672, 126.79999923706056, 5.21080493927002, -2.0, 19.200780868530277, -2.0, 2.7989999055862427, -2.0, 7.548500061035156, 5.127500057220459, -2.0, 3.1687283515930176, -2.0, 103.79999923706056, -2.0, -2.0, 99.67396926879884, 99.6057586669922, -2.0, -2.0, -2.0, 0.0213193036615848, -2.0, 138.5, -2.0, -2.0, 6.442499876022339, 268.3000030517578, 3.093000054359436, -2.0, -2.0, -2.0, 0.2710625678300857, 270.8000030517578, -2.0, -2.0, 142.0, -2.0, -2.0, 72.0999984741211, 69.0999984741211, -2.0,
                    -2.0, 93.5, 70.53973007202148, 269.5, 89.5, 0.0874376222491264, -2.0, 8.70550012588501, 3.162999987602234, -2.0, -2.0, -2.0, -2.0, -2.0, 6.624499797821045, -2.0, 12.648000240325928, 117.67363357543944, 2.712035059928894, 141.5, 110.60000228881836, -0.6580893993377686, 10.47849988937378, -2.0, -2.0, 114.8056755065918, -2.0, -2.0, 86.25765228271484, -2.0, -2.0, -2.31755793094635, -2.0, 86.58803939819336, -2.0, 107.2772102355957, 1.74628084897995, -2.0, -2.0, -2.0, 114.0999984741211, 74.58694076538086, -2.0, -2.0, -2.0, -2.0, -2.0, 85.24380493164062, 96.5, -2.0, -2.0, -2.0, 56.96506118774414, 69.5, 4.305999994277954, -2.0, 17.379805088043213, -2.0, 199.6000061035156,
                    47.03959465026856, -2.0, -2.0, -2.0, 150.5, 258.0, 65.5, -2.0, 0.905371904373169, -2.0, 6.629499912261963, 53.572513580322266, 80.5, 40.35166549682617, -2.0, -2.0, -2.0, -2.0, 1.455805003643036, -2.0, -2.0, 263.1000061035156, -2.0, -2.0, 10.739500045776367, 94.5, 5.650302648544312, 5.733548164367676, 79.5, 7.394500017166138, 5.276904106140137, -2.0, -2.0, 171.6999969482422, -2.0, -2.0, 5.200754165649414, -2.0, -2.0, -2.0, 153.5, -2.0, -2.0, -2.0, -2.0, 89.13872528076172, 84.80733489990234, 69.5, -2.0, 74.9739761352539, 283.0, 2.964853286743164, 255.0, -2.0, -2.0, -2.0, -2.0, 91.5, 3.067789912223816, -2.0, -2.0, -2.0, 9.1395001411438, 87.5, -2.0, -2.0, 85.59412002563477,
                    2.0690956115722656, -2.0, -2.0, -2.0, 66.5, 92.803955078125, -2.0, 80.0, 0.4120146632194519, -2.0, -2.0, 158.5, -2.0, 5.859499931335449, -2.0, -2.0, 76.70000076293945, 106.35651016235352, -2.0, 155.5, -2.0, -2.0, 91.5, 97.00950622558594, -2.3106133937835693, -2.0, 148.5, -2.0, 101.5, -2.0, -2.0, 113.2738265991211, -2.0, -2.0, 80.5, -2.0, 94.9725456237793, -2.0, 94.5, -2.0, -2.0, 79.5, 3.82099997997284, 65.021484375, -2.0, -2.0, 180.5, 74.5, 52.5, 191.3000030517578, -2.0, -2.0, 60.5, -2.0, 138.0999984741211, 28.7581205368042, 71.5, -2.0, -2.0, -2.0, 5.44840931892395, -2.0, 5.010929822921753, 167.5, -2.0, -2.0, -2.0, 220.5999984741211, 4.844145774841309, 6.923499822616577,
                    180.3000030517578, -2.0, -2.0, -2.0, 84.45674133300781, 20.66207981109619, -2.0, 9.037102699279783, 181.79999542236328, -2.0, 78.5, 179.5, -2.0, 29.900219917297363, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 4.75849986076355, 168.5, -2.0, -2.0, 8.001000165939331, 50.04906463623047, 7.632500171661377, 19.61096477508545, -2.0, 48.16451072692871, 5.688704490661621, -2.0, 5.415999889373779, -2.0, 5.873499870300293, 47.18850517272949, -2.0, -2.0, -2.0, -2.0, -2.0, 4.466421604156494, -2.0, 80.5, 176.5, -2.0, -2.0, -2.0, 3.5840693712234497, -2.0, -2.0];

const FEATURES = [1, 1, 1, -2, 2, 5, 0, 0, -2, -2, -2, -2, -2, 2, 5, 5, -2, -2, 5, 1, -2, 6, -2, -2, 0, -2, -2, -2, 4, 4, 5, 3, 0, 5, 2, -2, -2, 0, -2, -2, -2, 1, 4, 2, 4, 1, 0, -2, -2, 5, -2, -2, -2, 0, 1, -2, -2, -2, 1, 0, 1, -2, 6, 2, -2, -2, -2, -2, 5, 5, 5, -2, -2, -2, 0, -2, -2, 0, 4, -2, 2, 6, -2, -2, -2, 2, 6, -2, -2, -2, 1, 5, 3, 5, -2, -2, 6, -2, 5, 3, 5, -2, 3, 2, -2, -2, 5, -2, -2, -2, 6, -2, -2, 2, 5, 6, 1, -2, -2, 5, -2, -2, 5, -2, -2, 3, 5, 3, 3, -2, -2, -2, 2, -2, -2, 5, 1, 5, 0, 1, 2, -2, 3, -2, 3, -2, 5, -2, -2, 6, 2, 4, 2, -2, 4, -2, -2, -2, 3, -2, -2, 1, -2, -2, -2, 3, -2, 5, 0, 0, -2, -2, -2, 2, 5, 2, -2, -2, -2, -2, 5, 5, 0, -2, 0, 2, 0, 0, -2, -2, 5, -2, -2, -2, -2, -2,
                  2, -2, -2, 1, 2, -2, -2, -2, 5, -2, 1, 3, 4, 5, 0, 5, -2, -2, 0, 0, 2, -2, -2, -2, -2, 5, -2, -2, 3, -2, -2, -2, 4, -2, -2, 4, 1, 1, 6, 2, -2, 6, 5, -2, -2, 0, -2, -2, 5, 3, -2, -2, 2, 3, 5, -2, -2, 6, -2, -2, -2, 0, 0, 5, -2, 5, -2, 2, -2, 2, 2, -2, 6, -2, 0, -2, -2, 5, 5, -2, -2, -2, 6, -2, 4, -2, -2, 2, 0, 2, -2, -2, -2, 3, 0, -2, -2, 4, -2, -2, 0, 0, -2, -2, 1, 5, 0, 1, 3, -2, 2, 2, -2, -2, -2, -2, -2, 2, -2, 2, 5, 6, 4, 0, 3, 2, -2, -2, 5, -2, -2, 5, -2, -2, 3, -2, 5, -2, 5, 6, -2, -2, -2, 0, 5, -2, -2, -2, -2, -2, 5, 1, -2, -2, -2, 5, 1, 2, -2, 5, -2, 0, 5, -2, -2, -2, 4, 0, 0, -2, 3, -2, 2, 5, 1, 5, -2, -2, -2, -2, 3, -2, -2, 0, -2, -2, 2, 1, 6, 3, 1, 2, 6, -2, -2, 0,
                  -2, -2, 6, -2, -2, -2, 4, -2, -2, -2, -2, 5, 5, 1, -2, 5, 0, 3, 0, -2, -2, -2, -2, 1, 6, -2, -2, -2, 2, 1, -2, -2, 5, 6, -2, -2, -2, 1, 5, -2, 0, 3, -2, -2, 4, -2, 2, -2, -2, 0, 5, -2, 4, -2, -2, 1, 5, 3, -2, 4, -2, 0, -2, -2, 5, -2, -2, 0, -2, 5, -2, 1, -2, -2, 1, 2, 5, -2, -2, 4, 1, 1, 0, -2, -2, 1, -2, 0, 5, 1, -2, -2, -2, 6, -2, 3, 4, -2, -2, -2, 0, 3, 2, 0, -2, -2, -2, 5, 5, -2, 6, 0, -2, 1, 4, -2, 5, -2, -2, -2, -2, -2, -2, -2, 2, 4, -2, -2, 2, 5, 2, 5, -2, 5, 3, -2, 2, -2, 2, 5, -2, -2, -2, -2, -2, 3, -2, 1, 4, -2, -2, -2, 3, -2, -2];

const CHILDREN_LEFT = [1, 2, 3, -1, 5, 6, 7, 8, -1, -1, -1, -1, -1, 14, 15, 16, -1, -1, 19, 20, -1, 22, -1, -1, 25, -1, -1, -1, 29, 30, 31, 32, 33, 34, 35, -1, -1, 38, -1, -1, -1, 42, 43, 44, 45, 46, 47, -1, -1, 50, -1, -1, -1, 54, 55, -1, -1, -1, 59, 60, 61, -1, 63, 64, -1, -1, -1, -1, 69, 70, 71, -1, -1, -1, 75, -1, -1, 78, 79, -1, 81, 82, -1, -1, -1, 86, 87, -1, -1, -1, 91, 92, 93, 94, -1, -1, 97, -1, 99, 100, 101, -1, 103, 104, -1, -1, 107, -1, -1, -1, 111, -1, -1, 114, 115, 116, 117, -1, -1, 120, -1, -1, 123, -1, -1, 126, 127, 128, 129, -1, -1, -1, 133, -1, -1, 136, 137, 138, 139, 140, 141, -1, 143, -1, 145, -1, 147, -1, -1, 150, 151, 152, 153, -1, 155, -1, -1, -1, 159, -1, -1, 162, -1,
                       -1, -1, 166, -1, 168, 169, 170, -1, -1, -1, 174, 175, 176, -1, -1, -1, -1, 181, 182, 183, -1, 185, 186, 187, 188, -1, -1, 191, -1, -1, -1, -1, -1, 197, -1, -1, 200, 201, -1, -1, -1, 205, -1, 207, 208, 209, 210, 211, 212, -1, -1, 215, 216, 217, -1, -1, -1, -1, 222, -1, -1, 225, -1, -1, -1, 229, -1, -1, 232, 233, 234, 235, 236, -1, 238, 239, -1, -1, 242, -1, -1, 245, 246, -1, -1, 249, 250, 251, -1, -1, 254, -1, -1, -1, 258, 259, 260, -1, 262, -1, 264, -1, 266, 267, -1, 269, -1, 271, -1, -1, 274, 275, -1, -1, -1, 279, -1, 281, -1, -1, 284, 285, 286, -1, -1, -1, 290, 291, -1, -1, 294, -1, -1, 297, 298, -1, -1, 301, 302, 303, 304, 305, -1, 307, 308, -1, -1, -1, -1, -1, 314,
                       -1, 316, 317, 318, 319, 320, 321, 322, -1, -1, 325, -1, -1, 328, -1, -1, 331, -1, 333, -1, 335, 336, -1, -1, -1, 340, 341, -1, -1, -1, -1, -1, 347, 348, -1, -1, -1, 352, 353, 354, -1, 356, -1, 358, 359, -1, -1, -1, 363, 364, 365, -1, 367, -1, 369, 370, 371, 372, -1, -1, -1, -1, 377, -1, -1, 380, -1, -1, 383, 384, 385, 386, 387, 388, 389, -1, -1, 392, -1, -1, 395, -1, -1, -1, 399, -1, -1, -1, -1, 404, 405, 406, -1, 408, 409, 410, 411, -1, -1, -1, -1, 416, 417, -1, -1, -1, 421, 422, -1, -1, 425, 426, -1, -1, -1, 430, 431, -1, 433, 434, -1, -1, 437, -1, 439, -1, -1, 442, 443, -1, 445, -1, -1, 448, 449, 450, -1, 452, -1, 454, -1, -1, 457, -1, -1, 460, -1, 462, -1, 464, -1,
                       -1, 467, 468, 469, -1, -1, 472, 473, 474, 475, -1, -1, 478, -1, 480, 481, 482, -1, -1, -1, 486, -1, 488, 489, -1, -1, -1, 493, 494, 495, 496, -1, -1, -1, 500, 501, -1, 503, 504, -1, 506, 507, -1, 509, -1, -1, -1, -1, -1, -1, -1, 517, 518, -1, -1, 521, 522, 523, 524, -1, 526, 527, -1, 529, -1, 531, 532, -1, -1, -1, -1, -1, 538, -1, 540, 541, -1, -1, -1, 545, -1, -1];

const CHILDREN_RIGHT = [28, 13, 4, -1, 12, 11, 10, 9, -1, -1, -1, -1, -1, 27, 18, 17, -1, -1, 24, 21, -1, 23, -1, -1, 26, -1, -1, -1, 466, 231, 90, 41, 40, 37, 36, -1, -1, 39, -1, -1, -1, 77, 58, 53, 52, 49, 48, -1, -1, 51, -1, -1, -1, 57, 56, -1, -1, -1, 68, 67, 62, -1, 66, 65, -1, -1, -1, -1, 74, 73, 72, -1, -1, -1, 76, -1, -1, 85, 80, -1, 84, 83, -1, -1, -1, 89, 88, -1, -1, -1, 204, 113, 96, 95, -1, -1, 98, -1, 110, 109, 102, -1, 106, 105, -1, -1, 108, -1, -1, -1, 112, -1, -1, 125, 122, 119, 118, -1, -1, 121, -1, -1, 124, -1, -1, 135, 132, 131, 130, -1, -1, -1, 134, -1, -1, 199, 180, 165, 164, 149, 142, -1, 144, -1, 146, -1, 148, -1, -1, 161, 158, 157, 154, -1, 156, -1, -1, -1, 160, -1, -1,
                        163, -1, -1, -1, 167, -1, 173, 172, 171, -1, -1, -1, 179, 178, 177, -1, -1, -1, -1, 196, 195, 184, -1, 194, 193, 190, 189, -1, -1, 192, -1, -1, -1, -1, -1, 198, -1, -1, 203, 202, -1, -1, -1, 206, -1, 228, 227, 224, 221, 214, 213, -1, -1, 220, 219, 218, -1, -1, -1, -1, 223, -1, -1, 226, -1, -1, -1, 230, -1, -1, 351, 296, 257, 244, 237, -1, 241, 240, -1, -1, 243, -1, -1, 248, 247, -1, -1, 256, 253, 252, -1, -1, 255, -1, -1, -1, 283, 278, 261, -1, 263, -1, 265, -1, 273, 268, -1, 270, -1, 272, -1, -1, 277, 276, -1, -1, -1, 280, -1, 282, -1, -1, 289, 288, 287, -1, -1, -1, 293, 292, -1, -1, 295, -1, -1, 300, 299, -1, -1, 346, 313, 312, 311, 306, -1, 310, 309, -1, -1, -1, -1,
                        -1, 315, -1, 345, 344, 339, 330, 327, 324, 323, -1, -1, 326, -1, -1, 329, -1, -1, 332, -1, 334, -1, 338, 337, -1, -1, -1, 343, 342, -1, -1, -1, -1, -1, 350, 349, -1, -1, -1, 403, 362, 355, -1, 357, -1, 361, 360, -1, -1, -1, 382, 379, 366, -1, 368, -1, 376, 375, 374, 373, -1, -1, -1, -1, 378, -1, -1, 381, -1, -1, 402, 401, 398, 397, 394, 391, 390, -1, -1, 393, -1, -1, 396, -1, -1, -1, 400, -1, -1, -1, -1, 429, 420, 407, -1, 415, 414, 413, 412, -1, -1, -1, -1, 419, 418, -1, -1, -1, 424, 423, -1, -1, 428, 427, -1, -1, -1, 441, 432, -1, 436, 435, -1, -1, 438, -1, 440, -1, -1, 447, 444, -1, 446, -1, -1, 459, 456, 451, -1, 453, -1, 455, -1, -1, 458, -1, -1, 461, -1, 463, -1,
                        465, -1, -1, 516, 471, 470, -1, -1, 515, 492, 477, 476, -1, -1, 479, -1, 485, 484, 483, -1, -1, -1, 487, -1, 491, 490, -1, -1, -1, 514, 499, 498, 497, -1, -1, -1, 513, 502, -1, 512, 505, -1, 511, 508, -1, 510, -1, -1, -1, -1, -1, -1, -1, 520, 519, -1, -1, 544, 537, 536, 525, -1, 535, 528, -1, 530, -1, 534, 533, -1, -1, -1, -1, -1, 539, -1, 543, 542, -1, -1, -1, 546, -1, -1];

const VALUES = [2.16374269005848, 0.0, 0.0, 0.0, 0.1520467836257309, 0.47953216374269, 0.4327485380116959, 0.391812865497076, 0.5964912280701754, 0.0, 1.3099415204678362, 1.3099415204678362, 0.0994152046783625, 0.8888888888888888, 1.368421052631579, 0.9239766081871345, 1.2514619883040936, 0.8421052631578947, 1.590643274853801, 1.824561403508772, 1.4269005847953216, 2.1871345029239766, 2.760233918128655, 1.8538011695906431, 1.4152046783625731, 1.4269005847953216, 0.5146198830409356, 0.327485380116959, 2.3391812865497075, 2.2339181286549707, 1.7894736842105263, 1.6140350877192982, 1.1578947368421053, 1.2456140350877192, 1.456140350877193, 1.4152046783625731, 2.526315789473684, 0.9766081871345028,
                0.0, 1.0935672514619883, 0.0, 1.6374269005847952, 1.7076023391812865, 1.6257309941520468, 1.719298245614035, 1.5789473684210529, 0.7777777777777777, 0.672514619883041, 1.4853801169590644, 1.590643274853801, 1.6023391812865495, 0.8421052631578947, 1.824561403508772, 1.5380116959064327, 1.6374269005847952, 0.0, 1.6374269005847952, 1.3742690058479532, 1.8362573099415205, 2.0, 2.0, 0.2573099415204678, 2.0, 2.076023391812866, 2.2339181286549707, 2.0116959064327484, 1.894736842105263, 3.4853801169590644, 1.7543859649122806, 1.871345029239766, 1.7777777777777777, 0.9766081871345028, 1.7777777777777777, 2.064327485380117, 1.654970760233918, 1.7426900584795322, 1.5087719298245614,
                1.5087719298245614, 1.5555555555555556, 1.274853801169591, 1.590643274853801, 1.543859649122807, 1.2690058479532165, 1.5555555555555556, 1.6842105263157894, 1.3099415204678362, 1.3216374269005848, 1.4853801169590644, 1.274853801169591, 0.0, 1.976608187134503, 2.08187134502924, 1.9415204678362568, 1.3567251461988303, 2.0, 1.2573099415204678, 1.9649122807017545, 4.56140350877193, 1.95906432748538, 1.976608187134503, 1.9532163742690056, 2.0350877192982457, 1.9415204678362568, 1.9883040935672516, 4.4795321637426895, 1.9883040935672516, 1.8830409356725144, 1.912280701754386, 1.2046783625730997, 2.3157894736842106, 1.1578947368421053, 1.409356725146199, 0.0, 2.198830409356725,
                2.853801169590643, 3.2865497076023398, 3.7134502923976607, 2.432748538011696, 3.812865497076024, 3.0994152046783627, 2.16374269005848, 3.1345029239766085, 2.6608187134502925, 2.6666666666666665, 0.0, 2.175438596491228, 1.4152046783625731, 1.9415204678362568, 2.023391812865497, 2.4444444444444446, 1.906432748538012, 0.0, 0.9649122807017544, 1.391812865497076, 0.0701754385964912, 2.192982456140351, 2.2339181286549707, 2.473684210526316, 2.6198830409356724, 2.637426900584796, 3.175438596491228, 2.7953216374269005, 3.8596491228070176, 1.865497076023392, 3.906432748538012, 4.573099415204679, 3.789473684210527, 3.812865497076024, 0.7485380116959064, 2.538011695906433, 2.514619883040936,
                2.573099415204678, 2.7894736842105265, 2.888888888888889, 0.8888888888888888, 0.0, 2.9941520467836256, 2.514619883040936, 1.6842105263157894, 2.2339181286549707, 0.0, 4.023391812865497, 0.0, 4.058479532163743, 0.5029239766081871, 2.175438596491228, 3.812865497076024, 2.1403508771929824, 1.894736842105263, 1.719298245614035, 1.7309941520467835, 1.1403508771929824, 2.105263157894737, 2.309941520467836, 2.8304093567251463, 2.8304093567251463, 0.4561403508771929, 2.8304093567251463, 0.0701754385964912, 2.245614035087719, 2.175438596491228, 2.198830409356725, 2.175438596491228, 2.3508771929824563, 2.128654970760234, 2.175438596491228, 2.175438596491228, 2.08187134502924, 2.08187134502924,
                0.0, 2.2339181286549707, 2.175438596491228, 2.6666666666666665, 0.0, 1.9415204678362568, 3.0994152046783627, 2.011695906432749, 2.046783625730994, 1.2105263157894737, 1.894736842105263, 1.274853801169591, 1.415204678362573, 0.5146198830409356, 1.9298245614035088, 1.847953216374269, 1.5087719298245614, 1.871345029239766, 1.9298245614035088, 1.9298245614035088, 1.8830409356725144, 1.8596491228070176, 1.9824561403508767, 2.1345029239766085, 1.7953216374269003, 1.824561403508772, 1.5789473684210529, 1.672514619883041, 1.6900584795321638, 0.0, 1.0058479532163742, 1.8830409356725144, 1.9883040935672516, 2.1052631578947367, 1.8362573099415205, 2.0233918128654973, 2.046783625730994,
                1.7309941520467835, 0.9824561403508772, 1.6842105263157894, 1.350877192982456, 1.7309941520467835, 2.47953216374269, 2.210526315789474, 2.47953216374269, 2.9590643274853803, 2.467836257309941, 0.0, 2.502923976608187, 3.3684210526315788, 4.128654970760234, 2.584795321637426, 2.2690058479532165, 2.3859649122807016, 0.3976608187134502, 3.391812865497076, 3.988304093567252, 3.9941520467836256, 2.374269005847953, 3.3216374269005846, 3.3216374269005846, 3.4912280701754383, 4.011695906432749, 3.39766081871345, 2.888888888888889, 2.3976608187134505, 3.2748538011695905, 0.7953216374269005, 2.432748538011696, 2.3625730994152048, 2.3976608187134505, 1.824561403508772, 2.3976608187134505,
                2.736842105263158, 2.374269005847953, 1.8830409356725144, 2.3859649122807016, 2.421052631578948, 2.374269005847953, 2.701754385964912, 2.5497076023391814, 2.9941520467836256, 2.953216374269006, 3.766081871345029, 2.2690058479532165, 2.08187134502924, 2.128654970760234, 0.2339181286549707, 2.4444444444444446, 2.169590643274854, 0.7602339181286549, 2.210526315789474, 4.631578947368421, 2.192982456140351, 2.573099415204678, 2.5321637426900585, 2.538011695906433, 2.3391812865497075, 2.584795321637427, 1.847953216374269, 3.263157894736842, 2.912280701754386, 2.935672514619883, 0.2339181286549707, 4.754385964912281, 4.187134502923977, 5.076023391812866, 2.08187134502924, 2.538011695906433,
                2.654970760233918, 2.47953216374269, 2.046783625730994, 2.0701754385964914, 2.175438596491228, 2.198830409356725, 2.2690058479532165, 2.4269005847953213, 2.2339181286549707, 2.245614035087719, 3.0058479532163744, 2.2339181286549707, 1.2982456140350878, 2.0701754385964914, 1.5555555555555556, 2.0233918128654973, 1.847953216374269, 2.046783625730994, 2.046783625730994, 2.0350877192982457, 2.0233918128654973, 1.91812865497076, 1.9883040935672516, 2.432748538011696, 2.4444444444444446, 0.1169590643274853, 1.9415204678362568, 1.9883040935672516, 1.543859649122807, 1.7076023391812865, 1.807017543859649, 1.087719298245614, 2.046783625730994, 0.3508771929824561, 2.046783625730994, 2.0,
                2.0994152046783627, 2.409356725146199, 2.4444444444444446, 1.4502923976608186, 2.046783625730994, 2.175438596491228, 2.04093567251462, 2.175438596491228, 1.8362573099415205, 2.298245614035088, 2.3859649122807016, 0.0, 1.8362573099415205, 1.9532163742690056, 1.9532163742690056, 1.2982456140350878, 1.695906432748538, 2.56140350877193, 2.6432748538011697, 3.95906432748538, 2.39766081871345, 4.099415204678363, 1.871345029239766, 4.128654970760234, 4.140350877192983, 4.093567251461988, 4.491228070175438, 3.134502923976608, 2.6198830409356724, 2.538011695906433, 2.526315789473684, 2.7485380116959064, 2.502923976608187, 2.526315789473684, 2.374269005847953, 2.47953216374269, 2.514619883040936,
                2.6432748538011697, 3.976608187134503, 2.631578947368421, 2.374269005847953, 2.04093567251462, 1.9883040935672516, 2.2222222222222223, 1.7543859649122806, 3.6140350877192975, 4.906432748538012, 3.263157894736842, 2.6666666666666665, 2.672514619883041, 2.7134502923976607, 2.7134502923976607, 2.730994152046784, 2.9239766081871346, 2.9005847953216373, 2.8771929824561404, 4.052631578947368, 4.994152046783626, 1.9415204678362568, 5.035087719298246, 2.7134502923976607, 2.7134502923976607, 2.128654970760234, 2.526315789473684, 4.017543859649123, 1.8362573099415205, 4.046783625730995, 2.5964912280701755, 2.0116959064327484, 2.491228070175439, 2.3391812865497075, 2.421052631578948, 4.128654970760234,
                2.3976608187134505, 2.152046783625731, 2.2046783625730995, 2.1052631578947367, 2.111111111111111, 0.4210526315789473, 2.678362573099415, 1.590643274853801, 2.432748538011696, 2.456140350877193, 2.526315789473684, 2.3976608187134505, 2.16374269005848, 2.1052631578947367, 2.2339181286549707, 2.690058479532164, 2.175438596491228, 2.0350877192982457, 1.4619883040935673, 0.47953216374269, 1.543859649122807, 2.04093567251462, 2.5497076023391814, 2.783625730994152, 4.1988304093567255, 2.760233918128655, 3.812865497076024, 1.1929824561403508, 3.9649122807017534, 2.7485380116959064, 2.7485380116959064, 0.0, 0.0, 3.2046783625730995, 2.526315789473684, 2.573099415204678, 2.6198830409356724, 2.543859649122808,
                2.573099415204678, 2.456140350877193, 2.421052631578948, 2.467836257309941, 2.5964912280701755, 2.1052631578947367, 2.5964912280701755, 2.6198830409356724, 2.467836257309941, 0.0701754385964912, 2.497076023391813, 2.409356725146199, 2.421052631578948, 1.9532163742690056, 2.08187134502924, 2.502923976608187, 1.9883040935672516, 2.467836257309941, 1.9298245614035088, 2.0, 1.608187134502924, 4.105263157894737, 4.058479532163743, 2.450292397660819, 2.3684210526315788, 4.052631578947368, 4.058479532163743, 4.058479532163743, 4.046783625730995, 0.0, 4.292397660818714, 0.0, 4.046783625730995, 3.8596491228070176, 4.058479532163743, 3.9649122807017534, 4.105263157894737, 4.087719298245614, 5.578947368421052,
                3.91812865497076, 4.081871345029239, 4.274853801169591, 4.0701754385964914, 3.672514619883041, 0.5029239766081871, 3.719298245614035, 4.081871345029239, 4.4678362573099415, 4.456140350877193, 2.538011695906433, 1.929824561403509, 2.257309941520468, 0.0, 4.444444444444445, 4.538011695906433, 4.4678362573099415, 5.818713450292398, 4.456140350877193, 4.502923976608187, 4.409356725146199, 5.251461988304094, 4.760233918128655, 5.251461988304094, 4.245614035087719, 4.3918128654970765, 0.0, 6.128654970760234, 3.719298245614035, 6.146198830409357, 6.941520467836257, 0.0, 5.637426900584796, 1.91812865497076, 1.824561403508772, 3.5321637426900585, 5.666666666666666, 5.695906432748538, 5.4795321637426895,
                5.502923976608187, 6.643274853801169, 5.485380116959064, 5.491228070175438, 5.614035087719298, 5.2631578947368425, 5.5321637426900585, 5.076023391812866, 4.701754385964913, 4.707602339181287, 2.409356725146199, 5.461988304093568, 4.748538011695906, 3.95906432748538, 6.257309941520468, 4.923976608187134, 6.304093567251462, 3.3567251461988303, 3.345029239766082, 6.760233918128655, 6.304093567251462, 1.0526315789473684, 0.0, 3.0058479532163744];

var Slope;

class EstimatedPowerDataFieldView extends WatchUi.SimpleDataField {

    hidden var PowerField;

    function initialize() {
        SimpleDataField.initialize();

        self.PowerField = createField("estimated_power", POWER_RECORD_ID, FitContributor.DATA_TYPE_FLOAT, { :nativeNum=>POWER_NATIVE_NUM_RECORD_MESG, :mesgType=>FitContributor.MESG_TYPE_RECORD, :units=>POWER_UNITS });
    }

    function getParameter(paramName, defaultValue)
	{
	    var paramValue = Application.Properties.getValue(paramName);
	    if (paramValue == null) {
	      paramValue = defaultValue;
	      Application.Properties.setValue(paramName, defaultValue);
	    }

	    if (paramValue == null) { return 0; }
	    return paramValue;
	}

    function power_physics(info) {
        
        var BikeWeight = getParameter("BIKE_WEIGHT", 9.8);
        
        var m = userWeight + BikeWeight; // total mass
        
        var P_rr = C_rr * m * info.currentSpeed * g;
        var P_aero = 0.5 * rho * C_d * A * Math.pow(info.currentSpeed, 3);

        if (info has :slope) {Slope = info.slope;}
        else if (info has :current_slope) {Slope = info.current_slope;}
        else if (info has :grade) {Slope = info.grade;}
        else {Slope = 1;} // sbatti

        if (Slope == null) {Slope = 1;} // check sui null

        var slope_rad = Slope * (Math.PI / 180);
        var P_gr = m * g * info.currentSpeed * Math.sin(slope_rad);

        var P = (P_rr + P_aero + P_gr) / eta;

        var reading;
        if (P / userWeight <= 0) {
            reading = 0;
        } else {
            reading = P / userWeight;
        }

        return reading;
    }

    function compute(info) {

        if (info has :slope) {Slope = info.slope;}
        else if (info has :current_slope) {Slope = info.current_slope;}
        else if (info has :grade) {Slope = info.grade;}
        else {Slope = 1;} // sbatti

        if (Slope == null) {Slope = 1;} // check sui null

        var instance = [
            info.altitude,
            info.currentCadence / 2,
            info.currentSpeed,
            Slope,
            info.currentHeartRate,
            info.elapsedDistance / 1000,
            power_physics(info)
        ];

        var node = 0;  // start from the root
        
        while (CHILDREN_LEFT[node] != -1) {  // while not a leaf
            var feature_index = FEATURES[node];
            var threshold = THRESHOLDS[node];
            
            if (instance[feature_index] <= threshold) {
                node = CHILDREN_LEFT[node];
            } else {
                node = CHILDREN_RIGHT[node];
            }
        }
        
        // When a leaf is reached
        var EstimatedWatt = VALUES[node] * userWeight;
        return EstimatedWatt;
    }

    // // Set your layout here. Anytime the size of obscurity of
    // // the draw context is changed this will be called.
    // function onLayout(dc) {
        
    //     var width = dc.getWidth();
	// 	var height = dc.getHeight();
	// 	// var obsc = getObscurityFlags();

    //     // var obs_LEFT = obsc & 1<0;
	// 	// var obs_TOP = obsc & 1<1;
	// 	// var obs_RIGHT = obsc & 1<2;
	// 	// var obs_BOTT = obsc & 1<4;

    //     if (height >= 240){
    //        View.setLayout(Rez.Layouts.BigLayout1(dc));
    //    	}else if (height > 120){
	// 		if (width==height){View.setLayout(Rez.Layouts.MediumLayout1(dc));}
	// 		else {View.setLayout(Rez.Layouts.BigLayout2(dc));}
    //    	}else if (height > 82){
	// 		if (width==height){View.setLayout(Rez.Layouts.MediumLayout1(dc));}
	// 		else {View.setLayout(Rez.Layouts.MediumLayout2(dc));}
    //    	}else if (height > 69){View.setLayout(Rez.Layouts.SmallLayout(dc));}
	// 	else {View.setLayout(Rez.Layouts.MicroLayout(dc));}

    //     // var labelView = View.findDrawableById("label") as Toybox.WatchUi.Text;
    //     // var valueView = View.findDrawableById("value") as Toybox.WatchUi.Text;

    //     (View.findDrawableById("label") as Toybox.WatchUi.Text).setText("Slope");
    // }
    

    // // function compute(info) {
    // // }

    // // Display the value you computed here. This will be called
    // // once a second when the data field is visible.
    // function onUpdate(dc) {
    // // Set the background color
    //     (View.findDrawableById("Background") as Toybox.WatchUi.Text).setColor(getBackgroundColor());
    //     var value = View.findDrawableById("value") as Toybox.WatchUi.Text;
    //     var label = View.findDrawableById("label") as Toybox.WatchUi.Text;
	// 	var pc = View.findDrawableById("pc") as Toybox.WatchUi.Text;

	// 	pc.setText(POWER_UNITS);
    //     label.setColor(Graphics.COLOR_WHITE);

    //     // Set the foreground color
	// 	if (getBackgroundColor() == Graphics.COLOR_BLACK) {
    //         value.setColor(Graphics.COLOR_WHITE);
	// 		pc.setColor(Graphics.COLOR_WHITE);
    //     } else {
    //         value.setColor(Graphics.COLOR_BLACK);
	// 		pc.setColor(Graphics.COLOR_BLACK);
    //     }
	// 	self.PowerField = compute(self).getValue();

	// 	value.setText(PowerField.format("%.1f"));

    //     // Call parent's onUpdate(dc) to redraw the layout
    //     View.onUpdate(dc);
    // }
}