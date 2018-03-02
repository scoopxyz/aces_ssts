import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

"""
A simple control interface for sigmoid generation. Control points are on the
derivative of the curve, where indefinite integration results in a sigmoid.

There are 4 control points and 2 shapers. The 4 control points are:

* Head
* Shoulder
* Knee
* Toe

Which control their respective points on the sigmoid. The two shaper functions
control the behavior between the Toe and the Knee, and the Shoulder and Head
respectively.

* tk_exponent
* sh_exponent

These are currently simple exponential, but could be swapped with another
function with other characteristics, especially if you want the second
derivative curve to be smooth. 
"""


# linear interpolation
def lerp(a, b, t):
    return a + (b - a) * t


# helper class
class Coordinate(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def sigmoid_gradient(x, toe, knee, tk_exp, shoulder, head, sh_exp):
    """The sigmoid control point function

    All functions (once agreed on) could then by symbolically integrated, and
    the resultant integrated function could be used instead (with additional
    functions to maintain the additional needed constants).

    :param x: input value
    :param toe: toe control point (x,y)
    :param knee: knee control point (x,y)
    :param tk_exp: toe-knee exponent
    :param shoulder: shoulder control point (x,y)
    :param head: head control point (x,y)
    :param sh_exp: shoulder-head exponent
    :return: output derivative value
    """

    if toe.x <= x <= knee.x:

        return (pow(x / knee.x, tk_exp) + toe.y) * knee.y

    elif knee.x < x <= shoulder.x:

        return lerp(knee.y, shoulder.y, (x - knee.x) / (shoulder.x - knee.x))

    elif shoulder.x < x <= head.x:

        return (pow(
            ((head.x - shoulder.x) - (x - shoulder.x)) / (head.x - shoulder.x),
            sh_exp) * (shoulder.y - head.y)) + head.y

    else:
        return head.y


# PLOT ########################################################################

# helper function for plotting
def process(x_data, _knee, _knee_x, _shoulder, _shoulder_x, _k_exp, _s_exp):
    y_data = np.zeros(np.size(x_data))

    for idx, x in enumerate(x_data):
        toe = Coordinate({"x": 0, "y": 0})
        knee = Coordinate({"x": _knee_x, "y": _knee})
        tk_exp = _k_exp
        shoulder = Coordinate({"x": _shoulder_x, "y": _shoulder})
        head = Coordinate({"x": 3.0, "y": 0.05})
        sh_exp = _s_exp

        y_data[idx] = sigmoid_gradient(x, toe, knee, tk_exp, shoulder, head,
                                       sh_exp)
    return y_data


# helper function for plotting
def process_integrate(x_data, _knee, _knee_x, _shoulder, _shoulder_x, _k_exp, _s_exp):
    y_data = process(x_data, _knee, _knee_x, _shoulder, _shoulder_x, _k_exp, _s_exp)
    y_data_int = np.cumsum(y_data)

    return y_data_int


# initialize
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.4)

x_data = np.arange(0, 4, 1.0 / 100)
normalize_data = np.arange(0, 1.0, 1.0 / 100)

[line1] = ax.plot(x_data, process(x_data, 0.5, 1.0, 0.5, 2.0, 2.2, 2.2) * 200.0,
                  linewidth=2, color='blue')
[line2] = ax.plot(x_data, process_integrate(x_data, 0.5, 1.0, 0.5, 2.0, 2.2, 2.2),
                  linewidth=2, color='red')

# axis
ax.set_xlim([0, 4])
ax.set_ylim([0, 150])

# sliders
body_y_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
body_y_slider = Slider(body_y_ax, 'Body', 0.01, 1.0, valinit=0.5)

knee_x_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
knee_x_slider = Slider(knee_x_ax, 'Knee X', 0.0, 2.0, valinit=1.0)

shoulder_x_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
shoulder_x_slider = Slider(shoulder_x_ax, 'Shoulder X', 1.0, 3.0, valinit=2.0)

knee_exp_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
knee_exp_slider = Slider(knee_exp_ax, 'Knee Exp', 0.0, 10.0, valinit=2.2)

shoulder_exp_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
shoulder_exp_slider = Slider(shoulder_exp_ax, 'Shoulder Exp', 0.0, 10.0,
                             valinit=2.2)


# action
def sliders_on_changed(val):
    line1.set_ydata(process(x_data, body_y_slider.val, knee_x_slider.val, body_y_slider.val, shoulder_x_slider.val,
                            knee_exp_slider.val,
                            shoulder_exp_slider.val) * 200.0)
    line2.set_ydata(
        process_integrate(x_data, body_y_slider.val, knee_x_slider.val, body_y_slider.val, shoulder_x_slider.val,
                          knee_exp_slider.val, shoulder_exp_slider.val))
    fig.canvas.draw_idle()


# update
body_y_slider.on_changed(sliders_on_changed)
knee_x_slider.on_changed(sliders_on_changed)
shoulder_x_slider.on_changed(sliders_on_changed)
knee_exp_slider.on_changed(sliders_on_changed)
shoulder_exp_slider.on_changed(sliders_on_changed)

# plot
plt.show()
