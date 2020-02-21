import colorsys
import warnings
from typing import Sequence

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QImage, QPixmap, QColor, QIcon

from Orange.util import Enum, hex_to_color, color_to_hex

NAN_COLOR = (128, 128, 128)

__all__ = ["Palette", "IndexedPalette",
           "DiscretePalette", "LimitedDiscretePalette", "DiscretePalettes",
           "DefaultDiscretePalette", "DefaultDiscretePaletteName",
           "DefaultRGBColors", "Dark2Colors",
           "ContinuousPalette", "ContinuousPalettes", "BinnedContinuousPalette",
           "DefaultContinuousPalette", "DefaultContinuousPaletteName",
           "ColorIcon", "get_default_curve_colors", "patch_variable_colors",
           "NAN_COLOR"]


class Palette:
    """
    Base class for enumerable named categorized palettes used for visualization
    of discrete and numeric data

    Attributes:
        name (str): unique name
        friendly_name (str): name to be shown in user interfaces
        category (str): category for user interfaces
        palette (np.ndarray): palette; an array of shape (n, 3)
        nan_color (np.ndarray): an array of shape (1, 3) storing the colors used
            for missing values
        flags (Palette.Flags): flags describing palettes properties
            - ColorBlindSafe: palette is color-blind safe
            - Diverging: palette passes through some neutral color (white,
              black) which appears in the middle. For binned continuous
              palettes the pure neutral color does not need to appear in a bin
            - Discrete: palette contains a small number of colors, like
              palettes for discrete values and binned palettes
    """
    Flags = Enum("PaletteFlags",
                 dict(NoFlags=0,
                      ColorBlindSafe=1,
                      Diverging=2,
                      Discrete=4),
                 type=int,
                 qualname="Palette.Flags")
    NoFlags, ColorBlindSafe, Diverging, Discrete = Flags

    def __init__(self, friendly_name, name, palette, nan_color=NAN_COLOR,
                 *, category=None, flags=0):
        self.name = name
        self.friendly_name = friendly_name
        self.category = category or name.split("_")[0].title()
        self.palette = np.array(palette).astype(np.ubyte)
        self.nan_color = nan_color
        self.flags = flags

    # qcolors and qcolor_w_nan must not be cached because QColor is mutable
    # and may be modified by the caller (and is, in many widgets)
    @property
    def qcolors(self):
        """An array of QColors in the palette"""
        return np.array([QColor(*col) for col in self.palette])

    @property
    def qcolors_w_nan(self):
        """An array of QColors in the palette + the color for nan values"""
        return np.array([QColor(*col) for col in self.palette]
                        + [QColor(*self.nan_color)])

    def copy(self):
        return type(self)(self.friendly_name, self.name, self.palette.copy(),
                          self.nan_color,
                          category=self.category, flags=self.flags)


class IndexedPalette(Palette):
    def __len__(self):
        return len(self.palette)

    def __getitem__(self, x):
        if isinstance(x, (Sequence, np.ndarray)):
            return self.values_to_qcolors(x)
        elif isinstance(x, slice):
            return [QColor(*col) for col in self.palette.__getitem__(x)]
        else:
            return self.value_to_qcolor(x)


class DiscretePalette(IndexedPalette):
    def __init__(self, friendly_name, name, palette, nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.Discrete):
        super().__init__(friendly_name, name, palette, nan_color,
                         category=category, flags=flags)

    @classmethod
    def from_colors(cls, palette):
        """
        Create a palette from an (n x 3) array of ints in range (0, 255)
        """
        return cls("Custom", "Custom", palette)

    @staticmethod
    def _color_indices(x):
        x = np.asanyarray(x)
        nans = np.isnan(x)
        if np.any(nans):
            x = x.copy()
            x[nans] = -1
        return x.astype(int), nans

    def values_to_colors(self, x):
        """
        Map values x to colors; values may include nan's

        Args:
            x (np.ndarray): an array of values between 0 and len(palette) - 1

        Returns:
            An array of ubytes of shape (len(x), 3), representing RGB triplets
        """
        x, nans = self._color_indices(x)
        colors = self.palette[x]
        colors[nans, :] = self.nan_color
        return colors

    def values_to_qcolors(self, x):
        """
        Map values x to QColors; values may include nan's

        Args:
            x (np.ndarray): an array of values between 0 and len(palette) - 1

        Returns:
            An array of len(x) QColors
        """
        x, _ = self._color_indices(x)
        return self.qcolors_w_nan[x]

    def value_to_color(self, x):
        """
        Return an RGB triplet (as np.ndarray) corresponding to value x.
        x may also be nan.
        """
        if np.isnan(x):
            return self.nan_color
        return self.palette[int(x)]

    def value_to_qcolor(self, x):
        """
        Return a QColor corresponding to value x. x may also be nan.
        """
        color = self.nan_color if np.isnan(x) else self.palette[int(x)]
        return QColor(*color)


class LimitedDiscretePalette(DiscretePalette):
    """
    A palette derived from DiscretePalette that has the prescribed number of
    colors.

    Colors are taken from DefaultRGBColors (the default discrete palette),
    unless the desired number of colors is too large. In this case, colors
    are generated by making a circle around the HSV space.
    """

    def __init__(self, number_of_colors, nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.Discrete, force_hsv=False):
        if number_of_colors <= len(DefaultRGBColors) and not force_hsv:
            palette = DefaultRGBColors.palette[:number_of_colors]
        else:
            hues = np.linspace(0, 1, number_of_colors, endpoint=False)
            palette = 255 * np.array(
                [colorsys.hsv_to_rgb(h, 1, 1) for h in hues])
        super().__init__("custom", "custom", palette, nan_color,
                         category=category, flags=flags)


class ContinuousPalette(Palette):
    """
    Palette for continuous values
    """

    def __init__(self, friendly_name, name, palette, nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.NoFlags):
        super().__init__(
            friendly_name, name,
            np.asarray(palette, dtype=np.ubyte), nan_color,
            category=category, flags=flags)

    @staticmethod
    def _color_indices(x, low=None, high=None):
        x = np.asarray(x)
        nans = np.isnan(x)
        if np.all(nans):
            return np.full(len(x), -1), nans

        if low is None:
            low = np.nanmin(x)
        if high is None:
            high = np.nanmax(x)
        diff = high - low
        if diff == 0:
            x = np.full(len(x), 128)
        else:
            x = 255 * (x - low) / diff
            x = np.clip(x, 0, 255)
        x[nans] = -1
        return np.round(x).astype(int), nans

    def values_to_colors(self, x, low=None, high=None):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped
            low (float or None): minimal value; if None, determined from data
            high (float or None): maximal value; if None, determined from data

        Returns:
            an array of shape (len(x), 3) with RGB values for each point
        """
        x, nans = self._color_indices(x, low, high)
        colors = self.palette[x]
        colors[nans, :] = self.nan_color
        return colors

    def values_to_qcolors(self, x, low=None, high=None):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped
            low (float or None): minimal value; if None, determined from data
            high (float or None): maximal value; if None, determined from data

        Returns:
            an array of shape (len(x), ) with QColors for each point
        """
        x, _ = self._color_indices(x, low, high)
        return self.qcolors_w_nan[x]

    @staticmethod
    def _color_index(x, low=0, high=1):
        if np.isnan(x):
            return -1
        diff = high - low
        if diff == 0:
            return 128
        return int(np.clip(np.round(255 * (x - low) / diff), 0, 255))

    def value_to_color(self, x, low=0, high=1):
        """
        Return an RGB triplet (as np.ndarray) corresponding to value x.
        x may also be nan.
        """
        x = self._color_index(x, low, high)
        if x == -1:
            return NAN_COLOR
        return self.palette[x]

    def value_to_qcolor(self, x, low=0, high=1):
        """
        Return a QColor corresponding to value x. x may also be nan.
        """
        if np.isnan(x):
            color = self.nan_color
        else:
            x = self._color_index(x, low, high)
            color = self.palette[x]
        return QColor(*color)

    __getitem__ = value_to_qcolor

    def lookup_table(self, low=None, high=None):
        """
        A lookup table for this pallette.

        Arguments `low` and `high` (between 0 and 255) can be used to use
        just a part of palette.

        Args:
            low (float or None): minimal value
            high (float or None): maximal value

        Returns:
            an array of shape (255, 3) with RGB values
        """
        return self.values_to_colors(np.arange(256) / 256, low, high)

    def color_strip(self, strip_length, strip_width, orientation=Qt.Horizontal):
        """
        Return a pixmap of given dimensions to be used for legends.

        Args:
            strip_length (int): length of the strip; may be horizontal or vertical
            strip_width (int): width of the strip
            orientation: strip orientation

        Returns:
            QPixmap with a strip
        """
        points = np.linspace(0, 255, strip_length, dtype=np.uint8)
        section = self.palette[np.newaxis, points, :].astype(np.ubyte)
        img = np.vstack((section,) * strip_width)
        if orientation == Qt.Horizontal:
            width, height = strip_length, strip_width
        else:
            width, height = strip_width, strip_length
            img = img.swapaxes(0, 1)[::-1].copy()
        pad_width = (-img.strides[1]) % 4
        if pad_width:
            padding = np.zeros((img.shape[0], pad_width, 3), np.ubyte)
            img = np.hstack((img, padding))
        img = QImage(img, width, height, img.strides[0], QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        return img

    @classmethod
    def from_colors(cls, color1, color2, pass_through=False):
        """
        Deprecated constructor for tests and easier migration from
        Variable.color. Constructs a palette that goes from color1 to color2.

        pass_throug can be a color through which the palette will pass,
        or `True` to pass through black. Default is `False`.
        """
        if pass_through is True:
            colors = (color1, (0, 0, 0), color2)
            xf = [0, 127, 255]
        elif pass_through:
            assert isinstance(pass_through, tuple)
            colors = (color1, pass_through, color2)
            xf = [0, 127, 255]
        else:
            colors = (color1, color2)
            xf = [0, 255]
        name = repr(colors)
        friendly_name = "Custom"
        x = np.arange(256)
        r = np.interp(x, xf, np.array([color[0] for color in colors]))
        g = np.interp(x, xf, np.array([color[1] for color in colors]))
        b = np.interp(x, xf, np.array([color[2] for color in colors]))
        palette = np.vstack((r, g, b)).T
        return cls(friendly_name, name, palette,
                   flags=Palette.Diverging if pass_through else Palette.NoFlags)


class BinnedContinuousPalette(IndexedPalette):
    """
    Continuous palettes that bins values.

    Besides the derived attributes, it has an attribute `bins` (np.ndarray),
    which contains bin boundaries, including the lower and the higher
    boundary, which are essentially ignored.
    """

    def __init__(self, friendly_name, name, bin_colors, bins,
                 nan_color=NAN_COLOR,
                 *, category=None, flags=Palette.Discrete):
        super().__init__(friendly_name, name, bin_colors, nan_color,
                         category=category, flags=flags)
        self.bins = bins

    @classmethod
    def from_palette(cls, palette, bins):
        """
        Construct a `BinnedPalette` from `ContinuousPalette` by picking the
        colors representing the centers of the bins.

        If given a `BinnedPalette`, the constructor returns a copy.

        Args:
            palette (ContinuousPalette or BinnedPalette): original palette
            bins (np.ndarray): bin boundaries
        """
        if isinstance(palette, cls):
            # Note that this silently ignores bins. This is done on purpose
            # to let predefined binned palettes override bins. Plus, it is
            # generally impossible to compute a binned palette with different
            # bins.
            return palette.copy()
        if isinstance(palette, ContinuousPalette):
            assert len(bins) >= 2
            mids = (bins[:-1] + bins[1:]) / 2
            bin_colors = palette.values_to_colors(mids, bins[0], bins[-1])
            return cls(
                palette.friendly_name, palette.name, bin_colors, bins,
                palette.nan_color, category=palette.category,
                flags=palette.flags | Palette.Discrete)
        raise TypeError(f"can't create palette from '{type(palette).__name__}'")

    def _bin_indices(self, x):
        nans = np.isnan(x)
        binx = np.digitize(x, self.bins[1:-1], right=True)
        binx.clip(0, len(self.bins) - 1)
        binx[nans] = -1
        return binx, nans

    def values_to_colors(self, x):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped

        Returns:
            an array of shape (len(x), 3) with RGB values for each point
        """

        binx, nans = self._bin_indices(x)
        colors = self.palette[binx]
        colors[nans] = self.nan_color
        return colors

    def values_to_qcolors(self, x):
        """
        Return an array of colors assigned to given values by the palette

        Args:
            x (np.array): colors to be mapped

        Returns:
            an array of shape (len(x), ) with QColors for each point
        """
        binx, _ = self._bin_indices(x)
        return self.qcolors_w_nan[binx]

    def value_to_color(self, x):
        """
        Return an RGB triplet (as np.ndarray) corresponding to value x.
        x may also be nan.
        """
        return self.values_to_colors([x])[0]

    def value_to_qcolor(self, x):
        """
        Return a QColor corresponding to value x. x may also be nan.
        """
        if np.isnan(x):
            color = self.nan_color
        else:
            binx, _ = self._bin_indices([x])
            color = self.palette[binx[0]]
        return QColor(*color)

    def copy(self):
        return type(self)(self.friendly_name, self.name, self.palette.copy(),
                          self.bins.copy(), self.nan_color,
                          category=self.category, flags=self.flags)


DefaultRGBColors = DiscretePalette("Default", "Default", [
    [70, 190, 250], [237, 70, 47], [170, 242, 43], [245, 174, 50],
    [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 255],
    [0, 128, 255], [255, 223, 128], [127, 111, 64], [92, 46, 0],
    [0, 84, 0], [192, 192, 0], [0, 127, 127], [128, 0, 0],
    [127, 0, 127]])

Dark2Colors = DiscretePalette("Dark", "Dark", [
    [27, 158, 119], [217, 95, 2], [117, 112, 179], [231, 41, 138],
    [102, 166, 30], [230, 171, 2], [166, 118, 29], [102, 102, 102]])

DiscretePalettes = {
    "default": DefaultRGBColors,
    "dark": Dark2Colors
}

DefaultDiscretePaletteName = "default"
DefaultDiscretePalette = DiscretePalettes[DefaultDiscretePaletteName]

# pylint: disable=line-too-long
ContinuousPalettes = {
    'diverging_bwr_40_95_c42': ContinuousPalette(
        'Coolwarm', 'diverging_bwr_40_95_c42',
        [[33, 81, 219], [37, 82, 219], [42, 83, 219], [46, 84, 220], [49, 85, 220], [53, 86, 220], [56, 87, 220],
         [59, 88, 220], [62, 89, 221], [65, 91, 221], [67, 92, 221], [70, 93, 221], [72, 94, 221], [75, 95, 222],
         [77, 96, 222], [80, 97, 222], [82, 99, 222], [84, 100, 223], [86, 101, 223], [88, 102, 223], [90, 103, 223],
         [92, 104, 223], [94, 105, 224], [96, 107, 224], [98, 108, 224], [100, 109, 224], [102, 110, 224],
         [104, 111, 225], [105, 112, 225], [107, 114, 225], [109, 115, 225], [111, 116, 225], [112, 117, 226],
         [114, 118, 226], [116, 120, 226], [117, 121, 226], [119, 122, 226], [121, 123, 227], [122, 124, 227],
         [124, 126, 227], [125, 127, 227], [127, 128, 227], [129, 129, 228], [130, 130, 228], [132, 132, 228],
         [133, 133, 228], [135, 134, 228], [136, 135, 229], [138, 136, 229], [139, 138, 229], [141, 139, 229],
         [142, 140, 229], [143, 141, 229], [145, 143, 230], [146, 144, 230], [148, 145, 230], [149, 146, 230],
         [151, 148, 230], [152, 149, 231], [153, 150, 231], [155, 151, 231], [156, 153, 231], [158, 154, 231],
         [159, 155, 231], [160, 156, 232], [162, 158, 232], [163, 159, 232], [164, 160, 232], [166, 161, 232],
         [167, 163, 232], [168, 164, 233], [170, 165, 233], [171, 166, 233], [172, 168, 233], [174, 169, 233],
         [175, 170, 233], [176, 172, 234], [178, 173, 234], [179, 174, 234], [180, 175, 234], [181, 177, 234],
         [183, 178, 234], [184, 179, 235], [185, 180, 235], [187, 182, 235], [188, 183, 235], [189, 184, 235],
         [190, 186, 235], [192, 187, 235], [193, 188, 236], [194, 190, 236], [195, 191, 236], [197, 192, 236],
         [198, 193, 236], [199, 195, 236], [200, 196, 236], [202, 197, 237], [203, 199, 237], [204, 200, 237],
         [205, 201, 237], [207, 203, 237], [208, 204, 237], [209, 205, 237], [210, 207, 238], [211, 208, 238],
         [213, 209, 238], [214, 211, 238], [215, 212, 238], [216, 213, 238], [218, 215, 238], [219, 216, 238],
         [220, 217, 239], [221, 218, 239], [222, 220, 239], [224, 221, 239], [225, 222, 239], [226, 224, 239],
         [227, 225, 239], [228, 226, 239], [229, 227, 239], [230, 228, 238], [231, 229, 238], [232, 229, 238],
         [233, 230, 237], [234, 231, 237], [235, 231, 236], [236, 231, 235], [237, 231, 234], [238, 231, 233],
         [238, 231, 232], [239, 230, 231], [239, 230, 230], [240, 229, 228], [240, 228, 227], [241, 227, 225],
         [241, 226, 223], [241, 225, 221], [241, 224, 220], [241, 222, 218], [242, 221, 216], [242, 219, 214],
         [242, 218, 212], [242, 216, 210], [242, 215, 208], [242, 213, 206], [242, 212, 204], [242, 210, 203],
         [242, 209, 201], [242, 207, 199], [242, 206, 197], [242, 204, 195], [241, 203, 193], [241, 201, 191],
         [241, 199, 189], [241, 198, 187], [241, 196, 185], [241, 195, 183], [241, 193, 181], [241, 192, 180],
         [240, 190, 178], [240, 189, 176], [240, 187, 174], [240, 185, 172], [240, 184, 170], [240, 182, 168],
         [239, 181, 166], [239, 179, 165], [239, 178, 163], [239, 176, 161], [238, 175, 159], [238, 173, 157],
         [238, 172, 155], [238, 170, 153], [237, 168, 152], [237, 167, 150], [237, 165, 148], [237, 164, 146],
         [236, 162, 144], [236, 161, 142], [236, 159, 140], [235, 157, 139], [235, 156, 137], [235, 154, 135],
         [234, 153, 133], [234, 151, 131], [234, 150, 130], [233, 148, 128], [233, 147, 126], [232, 145, 124],
         [232, 143, 122], [232, 142, 121], [231, 140, 119], [231, 139, 117], [230, 137, 115], [230, 136, 113],
         [230, 134, 112], [229, 132, 110], [229, 131, 108], [228, 129, 106], [228, 128, 105], [227, 126, 103],
         [227, 124, 101], [226, 123, 99], [226, 121, 98], [225, 120, 96], [225, 118, 94], [224, 116, 92],
         [224, 115, 91], [223, 113, 89], [223, 111, 87], [222, 110, 85], [222, 108, 84], [221, 106, 82], [221, 105, 80],
         [220, 103, 79], [219, 102, 77], [219, 100, 75], [218, 98, 73], [218, 96, 72], [217, 95, 70], [216, 93, 68],
         [216, 91, 67], [215, 90, 65], [215, 88, 63], [214, 86, 62], [213, 84, 60], [213, 82, 58], [212, 81, 56],
         [211, 79, 55], [211, 77, 53], [210, 75, 51], [209, 73, 50], [209, 71, 48], [208, 69, 46], [207, 68, 45],
         [207, 66, 43], [206, 64, 41], [205, 61, 40], [205, 59, 38], [204, 57, 36], [203, 55, 34], [202, 53, 33],
         [202, 51, 31], [201, 48, 29], [200, 46, 27], [200, 43, 26], [199, 41, 24], [198, 38, 22], [197, 35, 20],
         [197, 32, 18], [196, 28, 16], [195, 25, 14], [194, 20, 12], [193, 15, 10], [193, 9, 8], [192, 2, 6]],
        flags=Palette.Diverging
    ),
    'diverging_gkr_60_10_c40': ContinuousPalette(
        'Green-Red', 'diverging_gkr_60_10_c40',
        [[54, 166, 22], [54, 165, 23], [54, 164, 23], [54, 162, 24], [54, 161, 24], [54, 160, 24], [54, 159, 25],
         [54, 158, 25], [54, 157, 26], [54, 155, 26], [54, 154, 26], [54, 153, 27], [54, 152, 27], [54, 151, 27],
         [54, 149, 27], [54, 148, 28], [54, 147, 28], [54, 146, 28], [54, 145, 28], [54, 144, 29], [54, 142, 29],
         [54, 141, 29], [54, 140, 29], [54, 139, 30], [54, 138, 30], [54, 137, 30], [54, 135, 30], [54, 134, 30],
         [53, 133, 30], [53, 132, 31], [53, 131, 31], [53, 130, 31], [53, 129, 31], [53, 127, 31], [53, 126, 31],
         [53, 125, 32], [53, 124, 32], [53, 123, 32], [53, 122, 32], [52, 121, 32], [52, 119, 32], [52, 118, 32],
         [52, 117, 32], [52, 116, 32], [52, 115, 32], [52, 114, 33], [51, 113, 33], [51, 112, 33], [51, 110, 33],
         [51, 109, 33], [51, 108, 33], [51, 107, 33], [51, 106, 33], [50, 105, 33], [50, 104, 33], [50, 103, 33],
         [50, 102, 33], [50, 100, 33], [49, 99, 33], [49, 98, 33], [49, 97, 33], [49, 96, 33], [49, 95, 33],
         [48, 94, 33], [48, 93, 33], [48, 92, 33], [48, 91, 33], [48, 90, 33], [47, 88, 33], [47, 87, 33], [47, 86, 33],
         [47, 85, 33], [46, 84, 33], [46, 83, 33], [46, 82, 33], [46, 81, 33], [45, 80, 33], [45, 79, 33], [45, 78, 33],
         [45, 77, 33], [44, 76, 33], [44, 75, 33], [44, 74, 33], [44, 73, 33], [43, 71, 33], [43, 70, 33], [43, 69, 33],
         [42, 68, 33], [42, 67, 32], [42, 66, 32], [42, 65, 32], [41, 64, 32], [41, 63, 32], [41, 62, 32], [40, 61, 32],
         [40, 60, 32], [40, 59, 32], [39, 58, 32], [39, 57, 32], [39, 56, 31], [38, 55, 31], [38, 54, 31], [38, 53, 31],
         [37, 52, 31], [37, 51, 31], [37, 50, 31], [36, 49, 31], [36, 48, 31], [36, 47, 30], [35, 46, 30], [35, 45, 30],
         [35, 44, 30], [34, 43, 30], [34, 42, 30], [34, 41, 30], [33, 40, 30], [33, 39, 29], [33, 39, 29], [32, 38, 29],
         [32, 37, 29], [32, 36, 29], [32, 35, 29], [32, 35, 29], [32, 34, 29], [33, 34, 29], [33, 33, 29], [33, 33, 29],
         [34, 32, 29], [35, 32, 29], [35, 32, 29], [36, 32, 29], [37, 32, 29], [38, 32, 29], [40, 32, 29], [41, 33, 30],
         [42, 33, 30], [44, 33, 30], [45, 34, 30], [47, 34, 30], [48, 34, 31], [50, 35, 31], [51, 35, 31], [53, 36, 31],
         [54, 36, 32], [56, 37, 32], [58, 37, 32], [59, 38, 32], [61, 38, 33], [62, 39, 33], [64, 39, 33], [66, 40, 33],
         [67, 40, 34], [69, 41, 34], [71, 41, 34], [72, 42, 34], [74, 42, 34], [75, 43, 35], [77, 43, 35], [79, 44, 35],
         [80, 45, 35], [82, 45, 36], [84, 46, 36], [85, 46, 36], [87, 46, 36], [89, 47, 37], [90, 47, 37], [92, 48, 37],
         [94, 48, 37], [95, 49, 38], [97, 49, 38], [99, 50, 38], [100, 50, 38], [102, 51, 39], [104, 51, 39],
         [106, 52, 39], [107, 52, 39], [109, 53, 39], [111, 53, 40], [112, 54, 40], [114, 54, 40], [116, 55, 40],
         [117, 55, 41], [119, 56, 41], [121, 56, 41], [123, 57, 41], [124, 57, 42], [126, 58, 42], [128, 58, 42],
         [130, 59, 42], [131, 59, 42], [133, 59, 43], [135, 60, 43], [137, 60, 43], [138, 61, 43], [140, 61, 44],
         [142, 62, 44], [144, 62, 44], [145, 63, 44], [147, 63, 44], [149, 64, 45], [151, 64, 45], [152, 65, 45],
         [154, 65, 45], [156, 65, 46], [158, 66, 46], [160, 66, 46], [161, 67, 46], [163, 67, 46], [165, 68, 47],
         [167, 68, 47], [169, 69, 47], [170, 69, 47], [172, 70, 47], [174, 70, 48], [176, 70, 48], [178, 71, 48],
         [180, 71, 48], [181, 72, 49], [183, 72, 49], [185, 73, 49], [187, 73, 49], [189, 73, 49], [191, 74, 50],
         [192, 74, 50], [194, 75, 50], [196, 75, 50], [198, 76, 50], [200, 76, 51], [202, 76, 51], [204, 77, 51],
         [205, 77, 51], [207, 78, 52], [209, 78, 52], [211, 79, 52], [213, 79, 52], [215, 79, 52], [217, 80, 53],
         [219, 80, 53], [220, 81, 53], [222, 81, 53], [224, 82, 53], [226, 82, 54], [228, 82, 54], [230, 83, 54],
         [232, 83, 54], [234, 84, 54], [236, 84, 55], [238, 85, 55], [240, 85, 55], [241, 85, 55], [243, 86, 55],
         [245, 86, 56], [247, 87, 56], [249, 87, 56], [251, 87, 56], [253, 88, 56]],
        flags=Palette.Diverging
    ),
    'linear_bgy_10_95_c74': ContinuousPalette(
        'Blue-Green-Yellow', 'linear_bgy_10_95_c74',
        [[0, 12, 125], [0, 13, 126], [0, 13, 128], [0, 14, 130], [0, 14, 132], [0, 15, 134], [0, 15, 136], [0, 16, 137],
         [0, 16, 139], [0, 17, 141], [0, 17, 143], [0, 18, 144], [0, 19, 146], [0, 19, 148], [0, 20, 150], [0, 21, 151],
         [0, 21, 153], [0, 22, 155], [0, 23, 156], [0, 23, 158], [0, 24, 159], [0, 25, 161], [0, 25, 162], [0, 26, 164],
         [0, 27, 165], [0, 28, 167], [0, 29, 168], [0, 29, 170], [0, 30, 171], [0, 31, 172], [0, 32, 174], [0, 33, 175],
         [0, 34, 176], [0, 35, 177], [0, 36, 179], [0, 36, 180], [0, 37, 181], [0, 38, 182], [0, 39, 183], [0, 41, 183],
         [0, 42, 184], [0, 43, 185], [0, 44, 186], [0, 45, 186], [0, 46, 187], [0, 47, 187], [0, 49, 187], [0, 50, 187],
         [0, 51, 188], [0, 52, 188], [0, 54, 188], [0, 55, 188], [0, 56, 188], [0, 57, 188], [0, 58, 188], [0, 60, 188],
         [0, 61, 187], [0, 62, 187], [0, 63, 187], [0, 64, 186], [0, 65, 186], [0, 67, 186], [0, 68, 185], [0, 69, 185],
         [0, 70, 184], [0, 71, 184], [0, 73, 183], [0, 74, 182], [0, 75, 181], [0, 76, 181], [0, 77, 180], [0, 78, 179],
         [0, 80, 178], [0, 81, 177], [0, 82, 176], [0, 83, 175], [0, 84, 174], [0, 86, 173], [0, 87, 172], [0, 88, 171],
         [0, 89, 169], [0, 90, 168], [0, 92, 167], [0, 93, 166], [0, 94, 165], [0, 95, 164], [0, 96, 162], [0, 97, 161],
         [0, 98, 160], [0, 100, 159], [0, 101, 158], [0, 102, 156], [0, 103, 155], [0, 104, 154], [0, 105, 153],
         [0, 107, 152], [0, 108, 150], [0, 109, 149], [0, 110, 148], [0, 111, 146], [0, 112, 145], [0, 113, 144],
         [0, 114, 143], [0, 116, 141], [0, 117, 140], [0, 118, 139], [0, 119, 137], [0, 120, 136], [0, 121, 134],
         [0, 122, 133], [0, 123, 131], [0, 125, 130], [0, 126, 128], [0, 127, 127], [0, 128, 125], [0, 129, 123],
         [0, 130, 122], [0, 131, 120], [0, 132, 118], [2, 133, 116], [6, 134, 114], [10, 135, 112], [13, 137, 110],
         [16, 138, 108], [19, 139, 106], [21, 140, 104], [23, 141, 102], [26, 142, 100], [28, 143, 98], [29, 144, 95],
         [31, 145, 93], [33, 146, 91], [34, 147, 88], [36, 148, 85], [37, 149, 83], [39, 150, 80], [40, 151, 78],
         [41, 152, 75], [42, 153, 72], [43, 155, 70], [44, 156, 67], [45, 157, 65], [45, 158, 62], [46, 159, 60],
         [46, 160, 58], [47, 161, 55], [47, 162, 53], [48, 163, 51], [48, 164, 49], [48, 165, 47], [48, 166, 45],
         [49, 167, 43], [49, 168, 41], [49, 169, 39], [49, 170, 37], [49, 171, 36], [49, 172, 34], [49, 173, 32],
         [49, 174, 31], [50, 175, 30], [50, 176, 29], [50, 177, 28], [50, 179, 27], [50, 180, 26], [50, 181, 26],
         [50, 182, 26], [50, 183, 26], [50, 184, 26], [51, 185, 26], [51, 186, 26], [51, 187, 26], [51, 188, 26],
         [51, 189, 26], [51, 190, 26], [51, 191, 26], [51, 192, 26], [52, 193, 25], [52, 194, 25], [52, 195, 25],
         [52, 196, 25], [52, 197, 25], [52, 198, 25], [52, 199, 25], [53, 200, 25], [53, 201, 25], [53, 202, 25],
         [53, 203, 25], [53, 204, 25], [53, 205, 25], [54, 206, 25], [54, 207, 25], [54, 208, 25], [54, 210, 25],
         [54, 211, 26], [54, 212, 26], [55, 213, 26], [55, 214, 26], [57, 215, 26], [59, 216, 26], [62, 216, 26],
         [65, 217, 26], [68, 218, 26], [72, 219, 26], [75, 220, 26], [79, 221, 26], [83, 221, 26], [87, 222, 26],
         [91, 223, 26], [95, 223, 26], [99, 224, 26], [103, 225, 27], [107, 225, 27], [111, 226, 27], [115, 227, 27],
         [119, 227, 27], [123, 228, 27], [127, 228, 27], [130, 229, 27], [134, 230, 27], [138, 230, 27], [142, 231, 28],
         [146, 231, 28], [150, 232, 28], [154, 232, 28], [157, 233, 28], [161, 233, 28], [165, 234, 28], [169, 234, 29],
         [172, 234, 29], [176, 235, 29], [180, 235, 29], [184, 236, 29], [187, 236, 29], [191, 236, 30], [195, 237, 30],
         [198, 237, 30], [202, 237, 30], [206, 238, 30], [209, 238, 31], [213, 238, 31], [217, 239, 31], [220, 239, 31],
         [224, 239, 32], [228, 239, 32], [231, 240, 32], [235, 240, 32], [238, 240, 33], [242, 240, 33], [246, 240, 33],
         [249, 241, 33], [253, 241, 34], [255, 241, 34], [255, 241, 34], [255, 241, 34], [255, 241, 35],
         [255, 241, 35]],
    ),
    'linear_bgy2_10_95_c74': ContinuousPalette(
        'Blue-Green-Yellow2', 'linear_bgy2_10_95_c74',
        [[5, 0, 172], [6, 2, 172], [6, 5, 172], [7, 8, 173], [7, 11, 173], [8, 14, 173], [8, 16, 174], [8, 18, 174],
         [9, 20, 174], [9, 22, 175], [10, 24, 175], [10, 26, 175], [10, 28, 176], [11, 29, 176], [11, 31, 176],
         [11, 32, 177], [12, 34, 177], [12, 35, 177], [12, 37, 177], [12, 38, 178], [13, 40, 178], [13, 41, 178],
         [13, 42, 178], [13, 44, 179], [13, 45, 179], [13, 46, 179], [13, 48, 179], [14, 49, 180], [14, 50, 180],
         [14, 51, 180], [14, 53, 180], [14, 54, 180], [14, 55, 180], [14, 56, 180], [14, 58, 181], [14, 59, 181],
         [14, 60, 181], [14, 61, 181], [14, 62, 181], [14, 64, 181], [13, 65, 181], [13, 66, 181], [13, 67, 181],
         [13, 68, 181], [13, 70, 181], [13, 71, 181], [13, 72, 181], [12, 73, 181], [12, 74, 181], [12, 76, 180],
         [12, 77, 180], [11, 78, 180], [11, 79, 180], [11, 80, 180], [10, 81, 179], [10, 83, 179], [10, 84, 179],
         [9, 85, 178], [9, 86, 178], [8, 87, 177], [8, 89, 177], [8, 90, 176], [7, 91, 175], [7, 92, 174], [6, 94, 174],
         [6, 95, 173], [6, 96, 172], [6, 97, 170], [6, 99, 169], [7, 100, 168], [8, 101, 167], [10, 102, 165],
         [12, 103, 164], [13, 104, 163], [15, 106, 161], [17, 107, 160], [19, 108, 158], [20, 109, 156], [22, 110, 155],
         [24, 111, 153], [25, 113, 152], [27, 114, 150], [29, 115, 148], [30, 116, 146], [31, 117, 145], [33, 118, 143],
         [34, 119, 141], [35, 120, 139], [36, 121, 137], [37, 122, 135], [38, 124, 133], [39, 125, 131], [40, 126, 129],
         [41, 127, 127], [42, 128, 125], [43, 129, 123], [44, 130, 121], [44, 131, 119], [45, 132, 117], [46, 133, 114],
         [46, 134, 112], [46, 135, 110], [47, 137, 108], [47, 138, 105], [48, 139, 103], [48, 140, 100], [48, 141, 98],
         [48, 142, 96], [48, 143, 93], [49, 144, 91], [49, 145, 89], [49, 146, 86], [50, 147, 84], [50, 148, 82],
         [51, 149, 80], [52, 150, 78], [52, 151, 76], [53, 152, 74], [54, 153, 72], [55, 154, 71], [56, 155, 69],
         [57, 156, 67], [58, 157, 65], [59, 157, 64], [60, 158, 62], [61, 159, 60], [62, 160, 59], [64, 161, 57],
         [65, 162, 55], [66, 163, 54], [68, 164, 52], [69, 164, 51], [71, 165, 49], [72, 166, 48], [74, 167, 46],
         [76, 168, 45], [77, 168, 44], [79, 169, 42], [81, 170, 41], [82, 171, 39], [84, 172, 38], [86, 172, 36],
         [88, 173, 35], [89, 174, 34], [91, 175, 32], [93, 175, 31], [95, 176, 30], [97, 177, 28], [99, 178, 27],
         [101, 178, 26], [103, 179, 24], [105, 180, 23], [107, 180, 22], [109, 181, 20], [111, 182, 19], [113, 182, 18],
         [115, 183, 16], [117, 184, 15], [119, 184, 14], [121, 185, 12], [123, 186, 11], [125, 186, 10], [127, 187, 8],
         [130, 188, 7], [132, 188, 6], [134, 189, 6], [136, 190, 5], [138, 190, 4], [140, 191, 4], [143, 191, 3],
         [145, 192, 3], [147, 192, 3], [149, 193, 2], [152, 194, 2], [154, 194, 2], [156, 195, 2], [158, 195, 2],
         [160, 196, 3], [162, 196, 3], [165, 197, 3], [167, 198, 3], [169, 198, 3], [171, 199, 4], [173, 199, 4],
         [175, 200, 4], [177, 200, 5], [179, 201, 5], [181, 202, 6], [183, 202, 6], [185, 203, 7], [187, 203, 8],
         [189, 204, 9], [191, 204, 10], [193, 205, 11], [194, 206, 12], [196, 206, 13], [198, 207, 14], [200, 207, 15],
         [202, 208, 16], [204, 209, 17], [205, 209, 18], [207, 210, 19], [209, 210, 20], [211, 211, 21], [212, 212, 23],
         [214, 212, 24], [216, 213, 25], [217, 214, 26], [219, 214, 28], [221, 215, 29], [222, 215, 30], [224, 216, 32],
         [225, 217, 33], [227, 218, 34], [228, 218, 36], [230, 219, 37], [231, 220, 39], [233, 220, 40], [234, 221, 42],
         [235, 222, 44], [236, 223, 45], [238, 223, 47], [239, 224, 49], [240, 225, 51], [241, 226, 52], [242, 227, 54],
         [243, 227, 56], [243, 228, 59], [244, 229, 61], [245, 230, 63], [245, 231, 66], [246, 232, 69], [246, 233, 72],
         [247, 234, 75], [248, 235, 79], [248, 236, 83], [249, 237, 88], [250, 237, 93], [250, 238, 98],
         [251, 239, 103], [252, 240, 109], [252, 241, 115], [253, 241, 121], [253, 242, 128], [254, 243, 135],
         [254, 243, 142], [255, 244, 150], [255, 245, 159], [255, 245, 168], [255, 246, 177], [255, 247, 187],
         [255, 247, 198], [254, 248, 210], [253, 248, 222], [251, 249, 235], [251, 249, 236]],
    ),
    'linear_viridis': ContinuousPalette(
        'Viridis', 'linear_viridis',
        [[68, 1, 84], [68, 2, 85], [68, 3, 87], [69, 5, 88], [69, 6, 90], [69, 8, 91], [70, 9, 92], [70, 11, 94], [70, 12, 95], [70, 14, 97], [71, 15, 98], [71, 17, 99], [71, 18, 101], [71, 20, 102], [71, 21, 103], [71, 22, 105], [71, 24, 106], [72, 25, 107], [72, 26, 108], [72, 28, 110], [72, 29, 111], [72, 30, 112], [72, 32, 113], [72, 33, 114], [72, 34, 115], [72, 35, 116], [71, 37, 117], [71, 38, 118], [71, 39, 119], [71, 40, 120], [71, 42, 121], [71, 43, 122], [71, 44, 123], [70, 45, 124], [70, 47, 124], [70, 48, 125], [70, 49, 126], [69, 50, 127], [69, 52, 127], [69, 53, 128], [69, 54, 129], [68, 55, 129], [68, 57, 130], [67, 58, 131], [67, 59, 131], [67, 60, 132], [66, 61, 132], [66, 62, 133], [66, 64, 133], [65, 65, 134], [65, 66, 134], [64, 67, 135], [64, 68, 135], [63, 69, 135], [63, 71, 136], [62, 72, 136], [62, 73, 137], [61, 74, 137], [61, 75, 137], [61, 76, 137], [60, 77, 138], [60, 78, 138], [59, 80, 138], [59, 81, 138], [58, 82, 139], [58, 83, 139], [57, 84, 139], [57, 85, 139], [56, 86, 139], [56, 87, 140], [55, 88, 140], [55, 89, 140], [54, 90, 140], [54, 91, 140], [53, 92, 140], [53, 93, 140], [52, 94, 141], [52, 95, 141], [51, 96, 141], [51, 97, 141], [50, 98, 141], [50, 99, 141], [49, 100, 141], [49, 101, 141], [49, 102, 141], [48, 103, 141], [48, 104, 141], [47, 105, 141], [47, 106, 141], [46, 107, 142], [46, 108, 142], [46, 109, 142], [45, 110, 142], [45, 111, 142], [44, 112, 142], [44, 113, 142], [44, 114, 142], [43, 115, 142], [43, 116, 142], [42, 117, 142], [42, 118, 142], [42, 119, 142], [41, 120, 142], [41, 121, 142], [40, 122, 142], [40, 122, 142], [40, 123, 142], [39, 124, 142], [39, 125, 142], [39, 126, 142], [38, 127, 142], [38, 128, 142], [38, 129, 142], [37, 130, 142], [37, 131, 141], [36, 132, 141], [36, 133, 141], [36, 134, 141], [35, 135, 141], [35, 136, 141], [35, 137, 141], [34, 137, 141], [34, 138, 141], [34, 139, 141], [33, 140, 141], [33, 141, 140], [33, 142, 140], [32, 143, 140], [32, 144, 140], [32, 145, 140], [31, 146, 140], [31, 147, 139], [31, 148, 139], [31, 149, 139], [31, 150, 139], [30, 151, 138], [30, 152, 138], [30, 153, 138], [30, 153, 138], [30, 154, 137], [30, 155, 137], [30, 156, 137], [30, 157, 136], [30, 158, 136], [30, 159, 136], [30, 160, 135], [31, 161, 135], [31, 162, 134], [31, 163, 134], [32, 164, 133], [32, 165, 133], [33, 166, 133], [33, 167, 132], [34, 167, 132], [35, 168, 131], [35, 169, 130], [36, 170, 130], [37, 171, 129], [38, 172, 129], [39, 173, 128], [40, 174, 127], [41, 175, 127], [42, 176, 126], [43, 177, 125], [44, 177, 125], [46, 178, 124], [47, 179, 123], [48, 180, 122], [50, 181, 122], [51, 182, 121], [53, 183, 120], [54, 184, 119], [56, 185, 118], [57, 185, 118], [59, 186, 117], [61, 187, 116], [62, 188, 115], [64, 189, 114], [66, 190, 113], [68, 190, 112], [69, 191, 111], [71, 192, 110], [73, 193, 109], [75, 194, 108], [77, 194, 107], [79, 195, 105], [81, 196, 104], [83, 197, 103], [85, 198, 102], [87, 198, 101], [89, 199, 100], [91, 200, 98], [94, 201, 97], [96, 201, 96], [98, 202, 95], [100, 203, 93], [103, 204, 92], [105, 204, 91], [107, 205, 89], [109, 206, 88], [112, 206, 86], [114, 207, 85], [116, 208, 84], [119, 208, 82], [121, 209, 81], [124, 210, 79], [126, 210, 78], [129, 211, 76], [131, 211, 75], [134, 212, 73], [136, 213, 71], [139, 213, 70], [141, 214, 68], [144, 214, 67], [146, 215, 65], [149, 215, 63], [151, 216, 62], [154, 216, 60], [157, 217, 58], [159, 217, 56], [162, 218, 55], [165, 218, 53], [167, 219, 51], [170, 219, 50], [173, 220, 48], [175, 220, 46], [178, 221, 44], [181, 221, 43], [183, 221, 41], [186, 222, 39], [189, 222, 38], [191, 223, 36], [194, 223, 34], [197, 223, 33], [199, 224, 31], [202, 224, 30], [205, 224, 29], [207, 225, 28], [210, 225, 27], [212, 225, 26], [215, 226, 25], [218, 226, 24], [220, 226, 24], [223, 227, 24], [225, 227, 24], [228, 227, 24], [231, 228, 25], [233, 228, 25], [236, 228, 26], [238, 229, 27], [241, 229, 28], [243, 229, 30], [246, 230, 31], [248, 230, 33], [250, 230, 34], [253, 231, 36]],
    ),
    'linear_cividis': ContinuousPalette(
        'Cividis', 'linear_cividis',
        [[0, 32, 76], [0, 32, 78], [0, 33, 80], [0, 34, 81], [0, 35, 83], [0, 35, 85], [0, 36, 86], [0, 37, 88], [0, 38, 90], [0, 38, 91], [0, 39, 93], [0, 40, 95], [0, 40, 97], [0, 41, 99], [0, 42, 100], [0, 42, 102], [0, 43, 104], [0, 44, 106], [0, 45, 108], [0, 45, 109], [0, 46, 110], [0, 46, 111], [0, 47, 111], [0, 47, 111], [0, 48, 111], [0, 49, 111], [0, 49, 111], [0, 50, 110], [0, 51, 110], [0, 52, 110], [0, 52, 110], [1, 53, 110], [6, 54, 110], [10, 55, 109], [14, 55, 109], [18, 56, 109], [21, 57, 109], [23, 57, 109], [26, 58, 108], [28, 59, 108], [30, 60, 108], [32, 60, 108], [34, 61, 108], [36, 62, 108], [38, 62, 108], [39, 63, 108], [41, 64, 107], [43, 65, 107], [44, 65, 107], [46, 66, 107], [47, 67, 107], [49, 68, 107], [50, 68, 107], [51, 69, 107], [53, 70, 107], [54, 70, 107], [55, 71, 107], [56, 72, 107], [58, 73, 107], [59, 73, 107], [60, 74, 107], [61, 75, 107], [62, 75, 107], [64, 76, 107], [65, 77, 107], [66, 78, 107], [67, 78, 107], [68, 79, 107], [69, 80, 107], [70, 80, 107], [71, 81, 107], [72, 82, 107], [73, 83, 107], [74, 83, 107], [75, 84, 107], [76, 85, 107], [77, 85, 107], [78, 86, 107], [79, 87, 108], [80, 88, 108], [81, 88, 108], [82, 89, 108], [83, 90, 108], [84, 90, 108], [85, 91, 108], [86, 92, 108], [87, 93, 109], [88, 93, 109], [89, 94, 109], [90, 95, 109], [91, 95, 109], [92, 96, 109], [93, 97, 110], [94, 98, 110], [95, 98, 110], [95, 99, 110], [96, 100, 110], [97, 101, 111], [98, 101, 111], [99, 102, 111], [100, 103, 111], [101, 103, 111], [102, 104, 112], [103, 105, 112], [104, 106, 112], [104, 106, 112], [105, 107, 113], [106, 108, 113], [107, 109, 113], [108, 109, 114], [109, 110, 114], [110, 111, 114], [111, 111, 114], [111, 112, 115], [112, 113, 115], [113, 114, 115], [114, 114, 116], [115, 115, 116], [116, 116, 117], [117, 117, 117], [117, 117, 117], [118, 118, 118], [119, 119, 118], [120, 120, 118], [121, 120, 119], [122, 121, 119], [123, 122, 119], [123, 123, 120], [124, 123, 120], [125, 124, 120], [126, 125, 120], [127, 126, 120], [128, 126, 120], [129, 127, 120], [130, 128, 120], [131, 129, 120], [132, 129, 120], [133, 130, 120], [134, 131, 120], [135, 132, 120], [136, 133, 120], [137, 133, 120], [138, 134, 120], [139, 135, 120], [140, 136, 120], [141, 136, 120], [142, 137, 120], [143, 138, 120], [144, 139, 120], [145, 140, 120], [146, 140, 120], [147, 141, 120], [148, 142, 120], [149, 143, 120], [150, 143, 119], [151, 144, 119], [152, 145, 119], [153, 146, 119], [154, 147, 119], [155, 147, 119], [156, 148, 119], [157, 149, 119], [158, 150, 118], [159, 151, 118], [160, 152, 118], [161, 152, 118], [162, 153, 118], [163, 154, 117], [164, 155, 117], [165, 156, 117], [166, 156, 117], [167, 157, 117], [168, 158, 116], [169, 159, 116], [170, 160, 116], [171, 161, 116], [172, 161, 115], [173, 162, 115], [174, 163, 115], [175, 164, 115], [176, 165, 114], [177, 166, 114], [178, 166, 114], [180, 167, 113], [181, 168, 113], [182, 169, 113], [183, 170, 112], [184, 171, 112], [185, 171, 112], [186, 172, 111], [187, 173, 111], [188, 174, 110], [189, 175, 110], [190, 176, 110], [191, 177, 109], [192, 177, 109], [193, 178, 108], [194, 179, 108], [195, 180, 108], [197, 181, 107], [198, 182, 107], [199, 183, 106], [200, 184, 106], [201, 184, 105], [202, 185, 105], [203, 186, 104], [204, 187, 104], [205, 188, 103], [206, 189, 103], [208, 190, 102], [209, 191, 102], [210, 192, 101], [211, 192, 101], [212, 193, 100], [213, 194, 99], [214, 195, 99], [215, 196, 98], [216, 197, 97], [217, 198, 97], [219, 199, 96], [220, 200, 96], [221, 201, 95], [222, 202, 94], [223, 203, 93], [224, 203, 93], [225, 204, 92], [227, 205, 91], [228, 206, 91], [229, 207, 90], [230, 208, 89], [231, 209, 88], [232, 210, 87], [233, 211, 86], [235, 212, 86], [236, 213, 85], [237, 214, 84], [238, 215, 83], [239, 216, 82], [240, 217, 81], [241, 218, 80], [243, 219, 79], [244, 220, 78], [245, 221, 77], [246, 222, 76], [247, 223, 75], [249, 224, 73], [250, 224, 72], [251, 225, 71], [252, 226, 70], [253, 227, 69], [255, 228, 67], [255, 229, 66], [255, 230, 66], [255, 231, 67], [255, 232, 68], [255, 233, 69]],
    ),
    'linear_inferno': ContinuousPalette(
        'Inferno', 'linear_inferno',
        [[0, 0, 3], [0, 0, 4], [0, 0, 6], [1, 0, 7], [1, 1, 9], [1, 1, 11], [2, 1, 14], [2, 2, 16], [3, 2, 18],
         [4, 3, 20], [4, 3, 22], [5, 4, 24], [6, 4, 27], [7, 5, 29], [8, 6, 31], [9, 6, 33], [10, 7, 35], [11, 7, 38],
         [13, 8, 40], [14, 8, 42], [15, 9, 45], [16, 9, 47], [18, 10, 50], [19, 10, 52], [20, 11, 54], [22, 11, 57],
         [23, 11, 59], [25, 11, 62], [26, 11, 64], [28, 12, 67], [29, 12, 69], [31, 12, 71], [32, 12, 74], [34, 11, 76],
         [36, 11, 78], [38, 11, 80], [39, 11, 82], [41, 11, 84], [43, 10, 86], [45, 10, 88], [46, 10, 90], [48, 10, 92],
         [50, 9, 93], [52, 9, 95], [53, 9, 96], [55, 9, 97], [57, 9, 98], [59, 9, 100], [60, 9, 101], [62, 9, 102],
         [64, 9, 102], [65, 9, 103], [67, 10, 104], [69, 10, 105], [70, 10, 105], [72, 11, 106], [74, 11, 106],
         [75, 12, 107], [77, 12, 107], [79, 13, 108], [80, 13, 108], [82, 14, 108], [83, 14, 109], [85, 15, 109],
         [87, 15, 109], [88, 16, 109], [90, 17, 109], [91, 17, 110], [93, 18, 110], [95, 18, 110], [96, 19, 110],
         [98, 20, 110], [99, 20, 110], [101, 21, 110], [102, 21, 110], [104, 22, 110], [106, 23, 110], [107, 23, 110],
         [109, 24, 110], [110, 24, 110], [112, 25, 110], [114, 25, 109], [115, 26, 109], [117, 27, 109], [118, 27, 109],
         [120, 28, 109], [122, 28, 109], [123, 29, 108], [125, 29, 108], [126, 30, 108], [128, 31, 107], [129, 31, 107],
         [131, 32, 107], [133, 32, 106], [134, 33, 106], [136, 33, 106], [137, 34, 105], [139, 34, 105], [141, 35, 105],
         [142, 36, 104], [144, 36, 104], [145, 37, 103], [147, 37, 103], [149, 38, 102], [150, 38, 102], [152, 39, 101],
         [153, 40, 100], [155, 40, 100], [156, 41, 99], [158, 41, 99], [160, 42, 98], [161, 43, 97], [163, 43, 97],
         [164, 44, 96], [166, 44, 95], [167, 45, 95], [169, 46, 94], [171, 46, 93], [172, 47, 92], [174, 48, 91],
         [175, 49, 91], [177, 49, 90], [178, 50, 89], [180, 51, 88], [181, 51, 87], [183, 52, 86], [184, 53, 86],
         [186, 54, 85], [187, 55, 84], [189, 55, 83], [190, 56, 82], [191, 57, 81], [193, 58, 80], [194, 59, 79],
         [196, 60, 78], [197, 61, 77], [199, 62, 76], [200, 62, 75], [201, 63, 74], [203, 64, 73], [204, 65, 72],
         [205, 66, 71], [207, 68, 70], [208, 69, 68], [209, 70, 67], [210, 71, 66], [212, 72, 65], [213, 73, 64],
         [214, 74, 63], [215, 75, 62], [217, 77, 61], [218, 78, 59], [219, 79, 58], [220, 80, 57], [221, 82, 56],
         [222, 83, 55], [223, 84, 54], [224, 86, 52], [226, 87, 51], [227, 88, 50], [228, 90, 49], [229, 91, 48],
         [230, 92, 46], [230, 94, 45], [231, 95, 44], [232, 97, 43], [233, 98, 42], [234, 100, 40], [235, 101, 39],
         [236, 103, 38], [237, 104, 37], [237, 106, 35], [238, 108, 34], [239, 109, 33], [240, 111, 31], [240, 112, 30],
         [241, 114, 29], [242, 116, 28], [242, 117, 26], [243, 119, 25], [243, 121, 24], [244, 122, 22], [245, 124, 21],
         [245, 126, 20], [246, 128, 18], [246, 129, 17], [247, 131, 16], [247, 133, 14], [248, 135, 13], [248, 136, 12],
         [248, 138, 11], [249, 140, 9], [249, 142, 8], [249, 144, 8], [250, 145, 7], [250, 147, 6], [250, 149, 6],
         [250, 151, 6], [251, 153, 6], [251, 155, 6], [251, 157, 6], [251, 158, 7], [251, 160, 7], [251, 162, 8],
         [251, 164, 10], [251, 166, 11], [251, 168, 13], [251, 170, 14], [251, 172, 16], [251, 174, 18], [251, 176, 20],
         [251, 177, 22], [251, 179, 24], [251, 181, 26], [251, 183, 28], [251, 185, 30], [250, 187, 33], [250, 189, 35],
         [250, 191, 37], [250, 193, 40], [249, 195, 42], [249, 197, 44], [249, 199, 47], [248, 201, 49], [248, 203, 52],
         [248, 205, 55], [247, 207, 58], [247, 209, 60], [246, 211, 63], [246, 213, 66], [245, 215, 69], [245, 217, 72],
         [244, 219, 75], [244, 220, 79], [243, 222, 82], [243, 224, 86], [243, 226, 89], [242, 228, 93], [242, 230, 96],
         [241, 232, 100], [241, 233, 104], [241, 235, 108], [241, 237, 112], [241, 238, 116], [241, 240, 121],
         [241, 242, 125], [242, 243, 129], [242, 244, 133], [243, 246, 137], [244, 247, 141], [245, 248, 145],
         [246, 250, 149], [247, 251, 153], [249, 252, 157], [250, 253, 160], [252, 254, 164]],
    ),
    'linear_magma': ContinuousPalette(
        'Magma', 'linear_magma',
        [[0, 0, 3], [0, 0, 4], [0, 0, 6], [1, 0, 7], [1, 1, 9], [1, 1, 11], [2, 2, 13], [2, 2, 15], [3, 3, 17],
         [4, 3, 19], [4, 4, 21], [5, 4, 23], [6, 5, 25], [7, 5, 27], [8, 6, 29], [9, 7, 31], [10, 7, 34], [11, 8, 36],
         [12, 9, 38], [13, 10, 40], [14, 10, 42], [15, 11, 44], [16, 12, 47], [17, 12, 49], [18, 13, 51], [20, 13, 53],
         [21, 14, 56], [22, 14, 58], [23, 15, 60], [24, 15, 63], [26, 16, 65], [27, 16, 68], [28, 16, 70], [30, 16, 73],
         [31, 17, 75], [32, 17, 77], [34, 17, 80], [35, 17, 82], [37, 17, 85], [38, 17, 87], [40, 17, 89], [42, 17, 92],
         [43, 17, 94], [45, 16, 96], [47, 16, 98], [48, 16, 101], [50, 16, 103], [52, 16, 104], [53, 15, 106],
         [55, 15, 108], [57, 15, 110], [59, 15, 111], [60, 15, 113], [62, 15, 114], [64, 15, 115], [66, 15, 116],
         [67, 15, 117], [69, 15, 118], [71, 15, 119], [72, 16, 120], [74, 16, 121], [75, 16, 121], [77, 17, 122],
         [79, 17, 123], [80, 18, 123], [82, 18, 124], [83, 19, 124], [85, 19, 125], [87, 20, 125], [88, 21, 126],
         [90, 21, 126], [91, 22, 126], [93, 23, 126], [94, 23, 127], [96, 24, 127], [97, 24, 127], [99, 25, 127],
         [101, 26, 128], [102, 26, 128], [104, 27, 128], [105, 28, 128], [107, 28, 128], [108, 29, 128], [110, 30, 129],
         [111, 30, 129], [113, 31, 129], [115, 31, 129], [116, 32, 129], [118, 33, 129], [119, 33, 129], [121, 34, 129],
         [122, 34, 129], [124, 35, 129], [126, 36, 129], [127, 36, 129], [129, 37, 129], [130, 37, 129], [132, 38, 129],
         [133, 38, 129], [135, 39, 129], [137, 40, 129], [138, 40, 129], [140, 41, 128], [141, 41, 128], [143, 42, 128],
         [145, 42, 128], [146, 43, 128], [148, 43, 128], [149, 44, 128], [151, 44, 127], [153, 45, 127], [154, 45, 127],
         [156, 46, 127], [158, 46, 126], [159, 47, 126], [161, 47, 126], [163, 48, 126], [164, 48, 125], [166, 49, 125],
         [167, 49, 125], [169, 50, 124], [171, 51, 124], [172, 51, 123], [174, 52, 123], [176, 52, 123], [177, 53, 122],
         [179, 53, 122], [181, 54, 121], [182, 54, 121], [184, 55, 120], [185, 55, 120], [187, 56, 119], [189, 57, 119],
         [190, 57, 118], [192, 58, 117], [194, 58, 117], [195, 59, 116], [197, 60, 116], [198, 60, 115], [200, 61, 114],
         [202, 62, 114], [203, 62, 113], [205, 63, 112], [206, 64, 112], [208, 65, 111], [209, 66, 110], [211, 66, 109],
         [212, 67, 109], [214, 68, 108], [215, 69, 107], [217, 70, 106], [218, 71, 105], [220, 72, 105], [221, 73, 104],
         [222, 74, 103], [224, 75, 102], [225, 76, 102], [226, 77, 101], [228, 78, 100], [229, 80, 99], [230, 81, 98],
         [231, 82, 98], [232, 84, 97], [234, 85, 96], [235, 86, 96], [236, 88, 95], [237, 89, 95], [238, 91, 94],
         [238, 93, 93], [239, 94, 93], [240, 96, 93], [241, 97, 92], [242, 99, 92], [243, 101, 92], [243, 103, 91],
         [244, 104, 91], [245, 106, 91], [245, 108, 91], [246, 110, 91], [246, 112, 91], [247, 113, 91], [247, 115, 92],
         [248, 117, 92], [248, 119, 92], [249, 121, 92], [249, 123, 93], [249, 125, 93], [250, 127, 94], [250, 128, 94],
         [250, 130, 95], [251, 132, 96], [251, 134, 96], [251, 136, 97], [251, 138, 98], [252, 140, 99], [252, 142, 99],
         [252, 144, 100], [252, 146, 101], [252, 147, 102], [253, 149, 103], [253, 151, 104], [253, 153, 105],
         [253, 155, 106], [253, 157, 107], [253, 159, 108], [253, 161, 110], [253, 162, 111], [253, 164, 112],
         [254, 166, 113], [254, 168, 115], [254, 170, 116], [254, 172, 117], [254, 174, 118], [254, 175, 120],
         [254, 177, 121], [254, 179, 123], [254, 181, 124], [254, 183, 125], [254, 185, 127], [254, 187, 128],
         [254, 188, 130], [254, 190, 131], [254, 192, 133], [254, 194, 134], [254, 196, 136], [254, 198, 137],
         [254, 199, 139], [254, 201, 141], [254, 203, 142], [253, 205, 144], [253, 207, 146], [253, 209, 147],
         [253, 210, 149], [253, 212, 151], [253, 214, 152], [253, 216, 154], [253, 218, 156], [253, 220, 157],
         [253, 221, 159], [253, 223, 161], [253, 225, 163], [252, 227, 165], [252, 229, 166], [252, 230, 168],
         [252, 232, 170], [252, 234, 172], [252, 236, 174], [252, 238, 176], [252, 240, 177], [252, 241, 179],
         [252, 243, 181], [252, 245, 183], [251, 247, 185], [251, 249, 187], [251, 250, 189], [251, 252, 191]] ,
    ),
    'linear_plasma': ContinuousPalette(
        'Plasma', 'linear_plasma',
[[12, 7, 134], [16, 7, 135], [19, 6, 137], [21, 6, 138], [24, 6, 139], [27, 6, 140], [29, 6, 141], [31, 5, 142], [33, 5, 143], [35, 5, 144], [37, 5, 145], [39, 5, 146], [41, 5, 147], [43, 5, 148], [45, 4, 148], [47, 4, 149], [49, 4, 150], [51, 4, 151], [52, 4, 152], [54, 4, 152], [56, 4, 153], [58, 4, 154], [59, 3, 154], [61, 3, 155], [63, 3, 156], [64, 3, 156], [66, 3, 157], [68, 3, 158], [69, 3, 158], [71, 2, 159], [73, 2, 159], [74, 2, 160], [76, 2, 161], [78, 2, 161], [79, 2, 162], [81, 1, 162], [82, 1, 163], [84, 1, 163], [86, 1, 163], [87, 1, 164], [89, 1, 164], [90, 0, 165], [92, 0, 165], [94, 0, 165], [95, 0, 166], [97, 0, 166], [98, 0, 166], [100, 0, 167], [101, 0, 167], [103, 0, 167], [104, 0, 167], [106, 0, 167], [108, 0, 168], [109, 0, 168], [111, 0, 168], [112, 0, 168], [114, 0, 168], [115, 0, 168], [117, 0, 168], [118, 1, 168], [120, 1, 168], [121, 1, 168], [123, 2, 168], [124, 2, 167], [126, 3, 167], [127, 3, 167], [129, 4, 167], [130, 4, 167], [132, 5, 166], [133, 6, 166], [134, 7, 166], [136, 7, 165], [137, 8, 165], [139, 9, 164], [140, 10, 164], [142, 12, 164], [143, 13, 163], [144, 14, 163], [146, 15, 162], [147, 16, 161], [149, 17, 161], [150, 18, 160], [151, 19, 160], [153, 20, 159], [154, 21, 158], [155, 23, 158], [157, 24, 157], [158, 25, 156], [159, 26, 155], [160, 27, 155], [162, 28, 154], [163, 29, 153], [164, 30, 152], [165, 31, 151], [167, 33, 151], [168, 34, 150], [169, 35, 149], [170, 36, 148], [172, 37, 147], [173, 38, 146], [174, 39, 145], [175, 40, 144], [176, 42, 143], [177, 43, 143], [178, 44, 142], [180, 45, 141], [181, 46, 140], [182, 47, 139], [183, 48, 138], [184, 50, 137], [185, 51, 136], [186, 52, 135], [187, 53, 134], [188, 54, 133], [189, 55, 132], [190, 56, 131], [191, 57, 130], [192, 59, 129], [193, 60, 128], [194, 61, 128], [195, 62, 127], [196, 63, 126], [197, 64, 125], [198, 65, 124], [199, 66, 123], [200, 68, 122], [201, 69, 121], [202, 70, 120], [203, 71, 119], [204, 72, 118], [205, 73, 117], [206, 74, 117], [207, 75, 116], [208, 77, 115], [209, 78, 114], [209, 79, 113], [210, 80, 112], [211, 81, 111], [212, 82, 110], [213, 83, 109], [214, 85, 109], [215, 86, 108], [215, 87, 107], [216, 88, 106], [217, 89, 105], [218, 90, 104], [219, 91, 103], [220, 93, 102], [220, 94, 102], [221, 95, 101], [222, 96, 100], [223, 97, 99], [223, 98, 98], [224, 100, 97], [225, 101, 96], [226, 102, 96], [227, 103, 95], [227, 104, 94], [228, 106, 93], [229, 107, 92], [229, 108, 91], [230, 109, 90], [231, 110, 90], [232, 112, 89], [232, 113, 88], [233, 114, 87], [234, 115, 86], [234, 116, 85], [235, 118, 84], [236, 119, 84], [236, 120, 83], [237, 121, 82], [237, 123, 81], [238, 124, 80], [239, 125, 79], [239, 126, 78], [240, 128, 77], [240, 129, 77], [241, 130, 76], [242, 132, 75], [242, 133, 74], [243, 134, 73], [243, 135, 72], [244, 137, 71], [244, 138, 71], [245, 139, 70], [245, 141, 69], [246, 142, 68], [246, 143, 67], [246, 145, 66], [247, 146, 65], [247, 147, 65], [248, 149, 64], [248, 150, 63], [248, 152, 62], [249, 153, 61], [249, 154, 60], [250, 156, 59], [250, 157, 58], [250, 159, 58], [250, 160, 57], [251, 162, 56], [251, 163, 55], [251, 164, 54], [252, 166, 53], [252, 167, 53], [252, 169, 52], [252, 170, 51], [252, 172, 50], [252, 173, 49], [253, 175, 49], [253, 176, 48], [253, 178, 47], [253, 179, 46], [253, 181, 45], [253, 182, 45], [253, 184, 44], [253, 185, 43], [253, 187, 43], [253, 188, 42], [253, 190, 41], [253, 192, 41], [253, 193, 40], [253, 195, 40], [253, 196, 39], [253, 198, 38], [252, 199, 38], [252, 201, 38], [252, 203, 37], [252, 204, 37], [252, 206, 37], [251, 208, 36], [251, 209, 36], [251, 211, 36], [250, 213, 36], [250, 214, 36], [250, 216, 36], [249, 217, 36], [249, 219, 36], [248, 221, 36], [248, 223, 36], [247, 224, 36], [247, 226, 37], [246, 228, 37], [246, 229, 37], [245, 231, 38], [245, 233, 38], [244, 234, 38], [243, 236, 38], [243, 238, 38], [242, 240, 38], [242, 241, 38], [241, 243, 38], [240, 245, 37], [240, 246, 35], [239, 248, 33]],
    ),
        'linear_bmy_10_95_c78': ContinuousPalette(
        'Blue-Magenta-Yellow', 'linear_bmy_10_95_c78',
        [[0, 12, 125], [0, 13, 126], [0, 13, 128], [0, 14, 130], [0, 14, 132], [0, 14, 134], [0, 15, 135], [0, 15, 137],
         [0, 16, 139], [0, 16, 140], [0, 17, 142], [0, 17, 144], [0, 17, 145], [0, 18, 147], [0, 18, 148], [0, 18, 150],
         [0, 19, 151], [0, 19, 153], [0, 19, 154], [0, 20, 155], [0, 20, 157], [0, 20, 158], [0, 20, 159], [0, 21, 160],
         [0, 21, 161], [0, 21, 162], [0, 21, 163], [0, 21, 164], [0, 21, 165], [0, 22, 166], [0, 22, 167], [0, 22, 167],
         [0, 22, 168], [0, 22, 169], [0, 22, 169], [0, 22, 169], [10, 22, 170], [21, 22, 170], [29, 21, 170],
         [35, 21, 170], [41, 21, 170], [47, 21, 169], [52, 20, 169], [57, 20, 168], [62, 19, 167], [67, 19, 166],
         [71, 18, 165], [76, 18, 164], [80, 17, 163], [83, 17, 162], [87, 16, 161], [90, 15, 160], [94, 15, 159],
         [97, 14, 159], [100, 14, 158], [103, 13, 157], [106, 13, 156], [108, 12, 155], [111, 11, 154], [114, 11, 153],
         [116, 10, 153], [119, 10, 152], [121, 9, 151], [124, 8, 151], [126, 8, 150], [128, 7, 149], [130, 7, 149],
         [133, 6, 148], [135, 6, 147], [137, 6, 147], [139, 5, 146], [141, 5, 146], [143, 4, 145], [145, 4, 145],
         [147, 4, 144], [149, 3, 144], [151, 3, 143], [153, 3, 143], [155, 2, 142], [157, 2, 142], [159, 2, 142],
         [161, 1, 141], [163, 1, 141], [165, 1, 140], [167, 1, 140], [169, 0, 139], [171, 0, 139], [172, 0, 138],
         [174, 0, 138], [176, 0, 137], [178, 0, 137], [180, 0, 136], [182, 0, 136], [184, 0, 135], [185, 0, 135],
         [187, 0, 135], [189, 0, 134], [191, 0, 134], [193, 0, 133], [195, 0, 133], [196, 0, 132], [198, 0, 132],
         [200, 0, 131], [202, 0, 131], [204, 0, 130], [206, 0, 130], [207, 0, 129], [209, 0, 129], [211, 0, 128],
         [213, 0, 128], [214, 0, 127], [216, 0, 127], [218, 0, 126], [219, 0, 126], [221, 0, 125], [222, 0, 124],
         [224, 1, 124], [226, 2, 123], [227, 4, 123], [229, 6, 122], [230, 8, 122], [232, 11, 121], [233, 13, 120],
         [234, 16, 120], [236, 18, 119], [237, 20, 119], [238, 22, 118], [240, 24, 117], [241, 26, 117], [242, 28, 116],
         [244, 30, 115], [245, 32, 115], [246, 34, 114], [247, 36, 113], [248, 38, 113], [249, 40, 112], [251, 42, 111],
         [252, 44, 111], [253, 46, 110], [254, 48, 109], [255, 50, 108], [255, 52, 108], [255, 54, 107], [255, 56, 106],
         [255, 58, 106], [255, 61, 105], [255, 63, 104], [255, 65, 103], [255, 67, 102], [255, 69, 102], [255, 71, 101],
         [255, 73, 100], [255, 75, 99], [255, 77, 98], [255, 80, 98], [255, 82, 97], [255, 84, 96], [255, 86, 95],
         [255, 88, 94], [255, 90, 93], [255, 93, 92], [255, 95, 91], [255, 97, 91], [255, 99, 90], [255, 101, 89],
         [255, 103, 88], [255, 106, 87], [255, 108, 86], [255, 110, 85], [255, 112, 84], [255, 114, 83], [255, 116, 82],
         [255, 118, 81], [255, 120, 80], [255, 122, 78], [255, 124, 77], [255, 126, 76], [255, 127, 75], [255, 129, 74],
         [255, 131, 73], [255, 133, 72], [255, 135, 71], [255, 136, 69], [255, 138, 68], [255, 140, 67], [255, 142, 66],
         [255, 143, 64], [255, 145, 63], [255, 147, 62], [255, 149, 60], [255, 150, 59], [255, 152, 57], [255, 154, 56],
         [255, 155, 54], [255, 157, 53], [255, 159, 51], [255, 160, 50], [255, 162, 48], [255, 163, 47], [255, 165, 46],
         [255, 166, 44], [255, 168, 43], [255, 169, 42], [255, 171, 41], [255, 172, 40], [255, 174, 39], [255, 175, 38],
         [255, 177, 38], [255, 178, 37], [255, 180, 36], [255, 181, 35], [255, 182, 35], [255, 184, 34], [255, 185, 34],
         [255, 187, 33], [255, 188, 32], [255, 189, 32], [255, 191, 31], [255, 192, 31], [255, 194, 31], [255, 195, 30],
         [255, 196, 30], [255, 198, 30], [255, 199, 29], [255, 200, 29], [255, 202, 29], [255, 203, 29], [255, 204, 29],
         [255, 206, 29], [255, 207, 28], [255, 208, 28], [255, 210, 28], [255, 211, 28], [255, 212, 28], [255, 214, 28],
         [255, 215, 29], [255, 216, 29], [255, 218, 29], [255, 219, 29], [255, 220, 29], [255, 222, 29], [255, 223, 30],
         [255, 224, 30], [255, 226, 30], [255, 227, 30], [255, 228, 31], [255, 230, 31], [255, 231, 31], [255, 232, 32],
         [255, 233, 32], [255, 235, 33], [255, 236, 33], [255, 237, 34], [255, 239, 34], [255, 240, 35],
         [255, 241, 35]],
    ),
    'linear_grey_10_95_c0': ContinuousPalette(
        'Dim gray', 'linear_grey_10_95_c0',
        [[27, 27, 27], [28, 28, 28], [29, 29, 29], [29, 29, 29], [30, 30, 30], [31, 31, 31], [31, 31, 31], [32, 32, 32],
         [33, 33, 33], [34, 34, 34], [34, 34, 34], [35, 35, 35], [36, 36, 36], [36, 36, 36], [37, 37, 37], [38, 38, 38],
         [38, 38, 38], [39, 39, 39], [40, 40, 40], [40, 40, 40], [41, 41, 41], [42, 42, 42], [43, 43, 43], [43, 43, 43],
         [44, 44, 44], [45, 45, 45], [45, 45, 45], [46, 46, 46], [47, 47, 47], [48, 48, 48], [48, 48, 48], [49, 49, 49],
         [50, 50, 50], [50, 50, 50], [51, 51, 51], [52, 52, 52], [53, 53, 53], [53, 53, 53], [54, 54, 54], [55, 55, 55],
         [56, 56, 56], [56, 56, 56], [57, 57, 57], [58, 58, 58], [59, 59, 59], [59, 59, 59], [60, 60, 60], [61, 61, 61],
         [62, 62, 62], [62, 62, 62], [63, 63, 63], [64, 64, 64], [65, 65, 65], [65, 65, 65], [66, 66, 66], [67, 67, 67],
         [68, 68, 68], [68, 68, 68], [69, 69, 69], [70, 70, 70], [71, 71, 71], [71, 71, 71], [72, 72, 72], [73, 73, 73],
         [74, 74, 74], [74, 74, 74], [75, 75, 75], [76, 76, 76], [77, 77, 77], [78, 78, 78], [78, 78, 78], [79, 79, 79],
         [80, 80, 80], [81, 81, 81], [81, 82, 82], [82, 82, 82], [83, 83, 83], [84, 84, 84], [85, 85, 85], [85, 85, 85],
         [86, 86, 86], [87, 87, 87], [88, 88, 88], [89, 89, 89], [89, 89, 89], [90, 90, 90], [91, 91, 91], [92, 92, 92],
         [93, 93, 93], [93, 93, 93], [94, 94, 94], [95, 95, 95], [96, 96, 96], [97, 97, 97], [97, 97, 97], [98, 98, 98],
         [99, 99, 99], [100, 100, 100], [101, 101, 101], [102, 102, 102], [102, 102, 102], [103, 103, 103],
         [104, 104, 104], [105, 105, 105], [106, 106, 106], [106, 106, 106], [107, 107, 107], [108, 108, 108],
         [109, 109, 109], [110, 110, 110], [111, 111, 111], [111, 111, 111], [112, 112, 112], [113, 113, 113],
         [114, 114, 114], [115, 115, 115], [116, 116, 116], [116, 116, 116], [117, 117, 117], [118, 118, 118],
         [119, 119, 119], [120, 120, 120], [121, 121, 121], [121, 121, 121], [122, 122, 122], [123, 123, 123],
         [124, 124, 124], [125, 125, 125], [126, 126, 126], [126, 127, 127], [127, 127, 127], [128, 128, 128],
         [129, 129, 129], [130, 130, 130], [131, 131, 131], [132, 132, 132], [132, 132, 132], [133, 133, 133],
         [134, 134, 134], [135, 135, 135], [136, 136, 136], [137, 137, 137], [138, 138, 138], [138, 138, 138],
         [139, 139, 139], [140, 140, 140], [141, 141, 141], [142, 142, 142], [143, 143, 143], [144, 144, 144],
         [145, 145, 145], [145, 145, 145], [146, 146, 146], [147, 147, 147], [148, 148, 148], [149, 149, 149],
         [150, 150, 150], [151, 151, 151], [152, 152, 152], [152, 152, 152], [153, 153, 153], [154, 154, 154],
         [155, 155, 155], [156, 156, 156], [157, 157, 157], [158, 158, 158], [159, 159, 159], [159, 159, 159],
         [160, 160, 160], [161, 161, 161], [162, 162, 162], [163, 163, 163], [164, 164, 164], [165, 165, 165],
         [166, 166, 166], [167, 167, 167], [167, 167, 167], [168, 168, 168], [169, 169, 169], [170, 170, 170],
         [171, 171, 171], [172, 172, 172], [173, 173, 173], [174, 174, 174], [175, 175, 175], [176, 176, 176],
         [176, 176, 176], [177, 177, 177], [178, 178, 178], [179, 179, 179], [180, 180, 180], [181, 181, 181],
         [182, 182, 182], [183, 183, 183], [184, 184, 184], [185, 185, 185], [185, 186, 186], [186, 186, 186],
         [187, 187, 187], [188, 188, 188], [189, 189, 189], [190, 190, 190], [191, 191, 191], [192, 192, 192],
         [193, 193, 193], [194, 194, 194], [195, 195, 195], [196, 196, 196], [196, 196, 196], [197, 197, 197],
         [198, 198, 198], [199, 199, 199], [200, 200, 200], [201, 201, 201], [202, 202, 202], [203, 203, 203],
         [204, 204, 204], [205, 205, 205], [206, 206, 206], [207, 207, 207], [208, 208, 208], [208, 209, 209],
         [209, 209, 209], [210, 210, 210], [211, 211, 211], [212, 212, 212], [213, 213, 213], [214, 214, 214],
         [215, 215, 215], [216, 216, 216], [217, 217, 217], [218, 218, 218], [219, 219, 219], [220, 220, 220],
         [221, 221, 221], [222, 222, 222], [223, 223, 223], [223, 224, 223], [224, 224, 224], [225, 225, 225],
         [226, 226, 226], [227, 227, 227], [228, 228, 228], [229, 229, 229], [230, 230, 230], [231, 231, 231],
         [232, 232, 232], [233, 233, 233], [234, 234, 234], [235, 235, 235], [236, 236, 236], [237, 237, 237],
         [238, 238, 238], [239, 239, 239], [240, 240, 240], [241, 241, 241]],
    ),
    'linear_kry_0_97_c73': ContinuousPalette(
        'Fire', 'linear_kry_0_97_c73',
        [[0, 0, 0], [7, 0, 0], [13, 0, 0], [18, 0, 0], [22, 0, 0], [25, 0, 0], [28, 0, 0], [31, 0, 0], [34, 0, 0],
         [36, 0, 0], [38, 0, 0], [40, 0, 0], [42, 0, 0], [44, 0, 0], [46, 0, 0], [48, 0, 0], [50, 0, 0], [51, 0, 0],
         [53, 0, 0], [54, 0, 0], [56, 0, 0], [57, 0, 0], [59, 0, 0], [60, 0, 0], [62, 0, 0], [63, 0, 0], [64, 0, 0],
         [66, 1, 0], [67, 1, 0], [69, 1, 0], [70, 1, 0], [72, 1, 0], [73, 1, 0], [75, 1, 0], [76, 1, 0], [78, 1, 0],
         [79, 1, 0], [81, 1, 0], [82, 1, 0], [84, 1, 0], [85, 1, 0], [87, 1, 0], [88, 1, 0], [90, 1, 0], [91, 1, 0],
         [93, 1, 0], [94, 1, 0], [96, 1, 0], [97, 1, 0], [99, 1, 0], [100, 1, 0], [102, 1, 0], [103, 1, 0], [105, 1, 0],
         [107, 2, 0], [108, 2, 0], [110, 2, 0], [111, 2, 0], [113, 2, 0], [114, 2, 0], [116, 2, 0], [118, 2, 0],
         [119, 2, 0], [121, 2, 0], [122, 2, 0], [124, 2, 0], [126, 2, 0], [127, 2, 0], [129, 2, 0], [131, 2, 0],
         [132, 3, 0], [134, 3, 0], [135, 3, 0], [137, 3, 0], [139, 3, 0], [140, 3, 0], [142, 3, 0], [144, 3, 0],
         [145, 3, 0], [147, 3, 0], [149, 3, 0], [150, 4, 0], [152, 4, 0], [154, 4, 0], [155, 4, 0], [157, 4, 0],
         [159, 4, 0], [160, 4, 0], [162, 4, 0], [164, 4, 0], [165, 5, 0], [167, 5, 0], [169, 5, 0], [171, 5, 0],
         [172, 5, 0], [174, 5, 0], [176, 6, 0], [177, 6, 0], [179, 6, 0], [181, 6, 0], [183, 6, 0], [184, 6, 0],
         [186, 7, 0], [188, 7, 0], [189, 7, 0], [191, 7, 0], [193, 7, 0], [195, 8, 0], [196, 8, 0], [198, 8, 0],
         [200, 8, 0], [202, 9, 0], [203, 9, 0], [205, 9, 0], [207, 9, 0], [209, 10, 0], [210, 10, 0], [212, 10, 0],
         [214, 11, 0], [216, 11, 0], [217, 12, 0], [219, 12, 0], [221, 12, 0], [223, 13, 0], [224, 13, 0], [226, 14, 0],
         [228, 15, 0], [230, 15, 0], [231, 16, 0], [233, 17, 0], [235, 18, 0], [236, 19, 0], [238, 21, 0], [239, 24, 0],
         [240, 26, 0], [241, 29, 0], [242, 33, 0], [243, 36, 0], [244, 39, 0], [244, 42, 0], [245, 46, 0], [246, 49, 0],
         [246, 52, 0], [247, 55, 0], [247, 58, 0], [247, 61, 0], [248, 63, 0], [248, 66, 0], [248, 69, 0], [249, 72, 0],
         [249, 74, 0], [249, 77, 0], [249, 79, 0], [250, 82, 0], [250, 84, 0], [250, 86, 0], [250, 89, 0], [250, 91, 0],
         [251, 93, 0], [251, 96, 0], [251, 98, 0], [251, 100, 0], [251, 102, 0], [251, 104, 0], [252, 106, 0],
         [252, 108, 0], [252, 110, 0], [252, 112, 0], [252, 114, 0], [252, 116, 0], [252, 118, 0], [252, 120, 0],
         [252, 122, 0], [252, 124, 0], [253, 126, 0], [253, 128, 0], [253, 130, 0], [253, 131, 0], [253, 133, 0],
         [253, 135, 0], [253, 137, 0], [253, 139, 0], [253, 140, 0], [253, 142, 0], [253, 144, 0], [253, 146, 0],
         [253, 147, 0], [254, 149, 0], [254, 151, 0], [254, 153, 0], [254, 154, 0], [254, 156, 0], [254, 158, 0],
         [254, 159, 0], [254, 161, 0], [254, 163, 0], [254, 164, 0], [254, 166, 0], [254, 168, 0], [254, 169, 0],
         [254, 171, 0], [254, 173, 0], [254, 174, 0], [254, 176, 0], [254, 177, 0], [254, 179, 0], [254, 181, 0],
         [254, 182, 0], [254, 184, 0], [254, 185, 0], [254, 187, 0], [254, 189, 0], [255, 190, 0], [255, 192, 0],
         [255, 193, 0], [255, 195, 0], [255, 196, 0], [255, 198, 0], [255, 199, 0], [255, 201, 0], [255, 203, 0],
         [255, 204, 0], [255, 206, 0], [255, 207, 0], [255, 209, 0], [255, 210, 0], [255, 212, 0], [255, 213, 0],
         [255, 215, 0], [255, 216, 0], [255, 218, 0], [255, 219, 0], [255, 221, 0], [255, 222, 0], [255, 224, 0],
         [255, 225, 0], [255, 227, 0], [255, 228, 0], [255, 230, 0], [255, 231, 0], [255, 233, 0], [255, 234, 0],
         [255, 236, 0], [255, 237, 0], [255, 239, 0], [255, 240, 0], [255, 242, 0], [255, 243, 0], [255, 245, 0],
         [255, 246, 0], [255, 248, 0], [255, 249, 0], [255, 251, 0], [255, 252, 0], [255, 254, 0], [255, 255, 0]],
    ),
    'diverging_protanopic_deuteranopic_bwy_60_95_c32': ContinuousPalette(
        'Diverging protanopic', 'diverging_protanopic_deuteranopic_bwy_60_95_c32',
        [[58, 144, 254], [62, 145, 254], [65, 146, 254], [68, 146, 254], [70, 147, 254], [73, 148, 254], [76, 148, 254],
         [78, 149, 254], [80, 150, 254], [83, 151, 253], [85, 151, 253], [87, 152, 253], [89, 153, 253], [91, 153, 253],
         [94, 154, 253], [96, 155, 253], [97, 155, 253], [99, 156, 253], [101, 157, 253], [103, 158, 253],
         [105, 158, 253], [107, 159, 253], [109, 160, 252], [110, 160, 252], [112, 161, 252], [114, 162, 252],
         [115, 163, 252], [117, 163, 252], [119, 164, 252], [120, 165, 252], [122, 165, 252], [124, 166, 252],
         [125, 167, 252], [127, 168, 251], [128, 168, 251], [130, 169, 251], [131, 170, 251], [133, 171, 251],
         [134, 171, 251], [136, 172, 251], [137, 173, 251], [139, 173, 251], [140, 174, 251], [141, 175, 251],
         [143, 176, 250], [144, 176, 250], [146, 177, 250], [147, 178, 250], [148, 179, 250], [150, 179, 250],
         [151, 180, 250], [152, 181, 250], [154, 182, 250], [155, 182, 250], [156, 183, 250], [158, 184, 249],
         [159, 185, 249], [160, 185, 249], [161, 186, 249], [163, 187, 249], [164, 187, 249], [165, 188, 249],
         [166, 189, 249], [168, 190, 249], [169, 191, 249], [170, 191, 248], [171, 192, 248], [173, 193, 248],
         [174, 194, 248], [175, 194, 248], [176, 195, 248], [177, 196, 248], [179, 197, 248], [180, 197, 248],
         [181, 198, 247], [182, 199, 247], [183, 200, 247], [185, 200, 247], [186, 201, 247], [187, 202, 247],
         [188, 203, 247], [189, 203, 247], [190, 204, 247], [192, 205, 246], [193, 206, 246], [194, 206, 246],
         [195, 207, 246], [196, 208, 246], [197, 209, 246], [198, 210, 246], [199, 210, 246], [201, 211, 246],
         [202, 212, 245], [203, 213, 245], [204, 213, 245], [205, 214, 245], [206, 215, 245], [207, 216, 245],
         [208, 217, 245], [209, 217, 245], [211, 218, 244], [212, 219, 244], [213, 220, 244], [214, 220, 244],
         [215, 221, 244], [216, 222, 244], [217, 223, 244], [218, 224, 244], [219, 224, 243], [220, 225, 243],
         [221, 226, 243], [222, 227, 243], [223, 228, 243], [224, 228, 243], [226, 229, 243], [227, 230, 242],
         [228, 231, 242], [229, 231, 242], [230, 232, 242], [231, 233, 242], [232, 234, 241], [233, 234, 241],
         [234, 235, 241], [234, 236, 240], [235, 236, 240], [236, 236, 239], [236, 237, 238], [237, 237, 237],
         [237, 237, 236], [238, 237, 235], [238, 236, 234], [238, 236, 232], [238, 236, 231], [238, 235, 229],
         [237, 234, 228], [237, 234, 226], [237, 233, 224], [236, 232, 223], [236, 231, 221], [236, 231, 219],
         [235, 230, 218], [235, 229, 216], [234, 228, 214], [234, 228, 213], [233, 227, 211], [233, 226, 209],
         [233, 225, 208], [232, 224, 206], [232, 224, 204], [231, 223, 202], [231, 222, 201], [230, 221, 199],
         [230, 220, 197], [229, 220, 196], [229, 219, 194], [228, 218, 192], [228, 217, 191], [227, 216, 189],
         [227, 216, 187], [226, 215, 186], [226, 214, 184], [226, 213, 182], [225, 213, 181], [225, 212, 179],
         [224, 211, 177], [224, 210, 176], [223, 209, 174], [223, 209, 172], [222, 208, 171], [222, 207, 169],
         [221, 206, 167], [220, 206, 166], [220, 205, 164], [219, 204, 162], [219, 203, 161], [218, 203, 159],
         [218, 202, 157], [217, 201, 156], [217, 200, 154], [216, 199, 152], [216, 199, 151], [215, 198, 149],
         [215, 197, 148], [214, 196, 146], [214, 196, 144], [213, 195, 143], [212, 194, 141], [212, 193, 139],
         [211, 193, 138], [211, 192, 136], [210, 191, 134], [210, 190, 133], [209, 190, 131], [208, 189, 129],
         [208, 188, 128], [207, 187, 126], [207, 187, 125], [206, 186, 123], [206, 185, 121], [205, 184, 120],
         [204, 184, 118], [204, 183, 116], [203, 182, 115], [203, 181, 113], [202, 181, 111], [201, 180, 110],
         [201, 179, 108], [200, 178, 106], [199, 178, 105], [199, 177, 103], [198, 176, 102], [198, 175, 100],
         [197, 175, 98], [196, 174, 97], [196, 173, 95], [195, 172, 93], [194, 172, 92], [194, 171, 90], [193, 170, 88],
         [193, 169, 87], [192, 169, 85], [191, 168, 83], [191, 167, 81], [190, 166, 80], [189, 166, 78], [189, 165, 76],
         [188, 164, 75], [187, 164, 73], [187, 163, 71], [186, 162, 69], [185, 161, 68], [185, 161, 66], [184, 160, 64],
         [183, 159, 62], [183, 159, 60], [182, 158, 59], [181, 157, 57], [180, 156, 55], [180, 156, 53], [179, 155, 51],
         [178, 154, 49], [178, 153, 47], [177, 153, 45], [176, 152, 43], [176, 151, 41], [175, 151, 39], [174, 150, 36],
         [173, 149, 34], [173, 149, 32], [172, 148, 29], [171, 147, 26], [171, 146, 23], [170, 146, 20], [169, 145, 17],
         [168, 144, 13], [168, 144, 8]],
        category="Color blind", flags=Palette.ColorBlindSafe | Palette.Diverging
    ),
    'diverging_tritanopic_cwr_75_98_c20': ContinuousPalette(
        'Diverging tritanopic', 'diverging_tritanopic_cwr_75_98_c20',
        [[41, 202, 231], [46, 202, 231], [50, 202, 231], [54, 203, 231], [57, 203, 231], [60, 203, 232], [64, 204, 232],
         [67, 204, 232], [70, 205, 232], [72, 205, 232], [75, 205, 232], [78, 206, 232], [80, 206, 233], [83, 207, 233],
         [85, 207, 233], [87, 207, 233], [89, 208, 233], [92, 208, 233], [94, 208, 234], [96, 209, 234], [98, 209, 234],
         [100, 210, 234], [102, 210, 234], [104, 210, 234], [106, 211, 234], [108, 211, 235], [110, 211, 235],
         [111, 212, 235], [113, 212, 235], [115, 213, 235], [117, 213, 235], [119, 213, 235], [120, 214, 236],
         [122, 214, 236], [124, 214, 236], [125, 215, 236], [127, 215, 236], [129, 216, 236], [130, 216, 236],
         [132, 216, 237], [134, 217, 237], [135, 217, 237], [137, 217, 237], [138, 218, 237], [140, 218, 237],
         [141, 219, 237], [143, 219, 238], [144, 219, 238], [146, 220, 238], [147, 220, 238], [149, 220, 238],
         [150, 221, 238], [152, 221, 238], [153, 222, 239], [155, 222, 239], [156, 222, 239], [158, 223, 239],
         [159, 223, 239], [160, 223, 239], [162, 224, 239], [163, 224, 240], [165, 225, 240], [166, 225, 240],
         [167, 225, 240], [169, 226, 240], [170, 226, 240], [172, 226, 240], [173, 227, 241], [174, 227, 241],
         [176, 228, 241], [177, 228, 241], [178, 228, 241], [180, 229, 241], [181, 229, 241], [182, 229, 242],
         [184, 230, 242], [185, 230, 242], [186, 230, 242], [188, 231, 242], [189, 231, 242], [190, 232, 242],
         [191, 232, 243], [193, 232, 243], [194, 233, 243], [195, 233, 243], [197, 233, 243], [198, 234, 243],
         [199, 234, 243], [200, 235, 244], [202, 235, 244], [203, 235, 244], [204, 236, 244], [205, 236, 244],
         [207, 236, 244], [208, 237, 244], [209, 237, 245], [210, 237, 245], [212, 238, 245], [213, 238, 245],
         [214, 239, 245], [215, 239, 245], [216, 239, 245], [218, 240, 246], [219, 240, 246], [220, 240, 246],
         [221, 241, 246], [223, 241, 246], [224, 241, 246], [225, 242, 246], [226, 242, 247], [227, 243, 247],
         [229, 243, 247], [230, 243, 247], [231, 244, 247], [232, 244, 247], [233, 244, 247], [235, 245, 247],
         [236, 245, 248], [237, 245, 248], [238, 246, 248], [239, 246, 248], [240, 246, 248], [242, 247, 248],
         [243, 247, 248], [244, 247, 248], [245, 247, 248], [246, 247, 247], [246, 247, 247], [247, 247, 247],
         [248, 246, 246], [248, 246, 246], [249, 246, 245], [249, 245, 244], [250, 245, 244], [250, 244, 243],
         [250, 243, 242], [250, 243, 241], [251, 242, 241], [251, 241, 240], [251, 241, 239], [251, 240, 238],
         [251, 239, 237], [251, 239, 237], [251, 238, 236], [252, 237, 235], [252, 237, 234], [252, 236, 233],
         [252, 235, 233], [252, 235, 232], [252, 234, 231], [252, 233, 230], [252, 232, 229], [252, 232, 229],
         [253, 231, 228], [253, 230, 227], [253, 230, 226], [253, 229, 225], [253, 228, 224], [253, 228, 224],
         [253, 227, 223], [253, 226, 222], [253, 226, 221], [253, 225, 220], [253, 224, 220], [254, 224, 219],
         [254, 223, 218], [254, 222, 217], [254, 221, 216], [254, 221, 216], [254, 220, 215], [254, 219, 214],
         [254, 219, 213], [254, 218, 212], [254, 217, 212], [254, 217, 211], [254, 216, 210], [254, 215, 209],
         [254, 215, 208], [254, 214, 208], [254, 213, 207], [255, 213, 206], [255, 212, 205], [255, 211, 204],
         [255, 210, 204], [255, 210, 203], [255, 209, 202], [255, 208, 201], [255, 208, 201], [255, 207, 200],
         [255, 206, 199], [255, 206, 198], [255, 205, 197], [255, 204, 197], [255, 204, 196], [255, 203, 195],
         [255, 202, 194], [255, 202, 193], [255, 201, 193], [255, 200, 192], [255, 199, 191], [255, 199, 190],
         [255, 198, 190], [255, 197, 189], [255, 197, 188], [255, 196, 187], [255, 195, 186], [255, 195, 186],
         [255, 194, 185], [255, 193, 184], [255, 193, 183], [255, 192, 182], [255, 191, 182], [255, 190, 181],
         [255, 190, 180], [255, 189, 179], [255, 188, 179], [255, 188, 178], [255, 187, 177], [255, 186, 176],
         [255, 186, 175], [255, 185, 175], [255, 184, 174], [255, 184, 173], [255, 183, 172], [255, 182, 172],
         [255, 181, 171], [255, 181, 170], [255, 180, 169], [255, 179, 169], [254, 179, 168], [254, 178, 167],
         [254, 177, 166], [254, 177, 165], [254, 176, 165], [254, 175, 164], [254, 174, 163], [254, 174, 162],
         [254, 173, 162], [254, 172, 161], [254, 172, 160], [254, 171, 159], [254, 170, 159], [254, 170, 158],
         [254, 169, 157], [254, 168, 156], [254, 167, 156], [254, 167, 155], [253, 166, 154], [253, 165, 153],
         [253, 165, 153], [253, 164, 152], [253, 163, 151], [253, 163, 150], [253, 162, 150], [253, 161, 149],
         [253, 160, 148]],
        category="Color blind", flags=Palette.ColorBlindSafe | Palette.Diverging
    ),
    'linear_protanopic_deuteranopic_kbw_5_98_c40': ContinuousPalette(
        'Linear protanopic', 'linear_protanopic_deuteranopic_kbw_5_98_c40',
        [[17, 17, 17], [17, 18, 19], [18, 19, 21], [19, 19, 23], [19, 20, 24], [20, 21, 26], [20, 22, 28], [20, 23, 29],
         [21, 23, 31], [21, 24, 33], [21, 25, 34], [22, 25, 36], [22, 26, 38], [22, 27, 39], [22, 27, 41], [22, 28, 43],
         [22, 29, 45], [22, 30, 46], [23, 30, 48], [23, 31, 50], [23, 32, 52], [23, 33, 54], [23, 33, 55], [23, 34, 57],
         [23, 35, 59], [22, 36, 61], [22, 36, 63], [22, 37, 64], [22, 38, 66], [22, 39, 68], [22, 39, 70], [21, 40, 72],
         [21, 41, 74], [21, 42, 75], [20, 43, 77], [20, 43, 79], [20, 44, 81], [19, 45, 83], [19, 46, 84], [19, 46, 86],
         [18, 47, 88], [18, 48, 90], [18, 49, 91], [17, 50, 93], [17, 50, 95], [17, 51, 96], [16, 52, 98], [16, 53, 99],
         [16, 54, 101], [16, 54, 103], [16, 55, 104], [16, 56, 106], [16, 57, 107], [15, 58, 109], [15, 59, 110],
         [15, 59, 112], [15, 60, 113], [15, 61, 115], [15, 62, 116], [15, 63, 118], [15, 63, 119], [15, 64, 121],
         [15, 65, 122], [15, 66, 124], [15, 67, 126], [15, 68, 127], [15, 68, 129], [15, 69, 130], [15, 70, 132],
         [15, 71, 133], [15, 72, 135], [15, 73, 136], [15, 73, 138], [15, 74, 139], [15, 75, 141], [15, 76, 142],
         [15, 77, 144], [15, 78, 146], [15, 79, 147], [15, 79, 149], [14, 80, 150], [14, 81, 152], [14, 82, 153],
         [14, 83, 155], [14, 84, 157], [14, 85, 158], [14, 86, 160], [14, 86, 161], [13, 87, 163], [13, 88, 165],
         [13, 89, 166], [13, 90, 168], [13, 91, 169], [13, 92, 171], [12, 92, 173], [12, 93, 174], [12, 94, 176],
         [12, 95, 178], [11, 96, 179], [11, 97, 181], [11, 98, 182], [10, 99, 184], [10, 100, 186], [10, 100, 187],
         [9, 101, 189], [9, 102, 191], [9, 103, 192], [8, 104, 194], [8, 105, 196], [7, 106, 197], [7, 107, 199],
         [7, 108, 201], [6, 109, 202], [6, 109, 204], [5, 110, 206], [5, 111, 207], [4, 112, 209], [4, 113, 211],
         [3, 114, 212], [3, 115, 214], [2, 116, 216], [2, 117, 217], [1, 118, 219], [1, 119, 221], [0, 120, 222],
         [0, 120, 224], [0, 121, 226], [1, 122, 227], [1, 123, 229], [2, 124, 230], [3, 125, 232], [5, 126, 233],
         [7, 127, 235], [11, 128, 236], [14, 129, 237], [18, 130, 238], [22, 131, 239], [26, 132, 240], [31, 132, 241],
         [35, 133, 242], [40, 134, 242], [45, 135, 242], [49, 136, 242], [54, 137, 242], [59, 138, 242], [64, 139, 241],
         [69, 140, 240], [74, 141, 239], [79, 141, 238], [84, 142, 236], [89, 143, 234], [93, 144, 232], [98, 145, 230],
         [103, 146, 228], [107, 147, 226], [111, 148, 223], [115, 149, 220], [119, 150, 217], [123, 151, 214],
         [127, 152, 212], [131, 153, 208], [134, 154, 205], [137, 155, 202], [141, 155, 199], [144, 156, 196],
         [147, 157, 193], [150, 158, 189], [152, 159, 186], [155, 160, 183], [158, 161, 180], [160, 162, 177],
         [163, 163, 174], [165, 164, 170], [168, 165, 167], [170, 166, 164], [172, 167, 161], [175, 168, 158],
         [177, 169, 155], [179, 170, 152], [181, 171, 149], [183, 172, 146], [185, 173, 143], [187, 174, 140],
         [189, 175, 137], [191, 176, 134], [193, 177, 131], [195, 178, 128], [197, 179, 125], [199, 180, 122],
         [201, 181, 119], [202, 182, 116], [204, 183, 113], [206, 184, 109], [208, 184, 106], [209, 185, 103],
         [211, 186, 100], [213, 187, 96], [214, 188, 93], [216, 189, 90], [218, 190, 86], [219, 191, 82],
         [221, 192, 79], [222, 193, 75], [224, 194, 72], [225, 195, 68], [227, 196, 64], [228, 197, 60], [229, 198, 57],
         [231, 199, 53], [232, 200, 49], [233, 201, 46], [235, 202, 42], [236, 203, 39], [237, 204, 37], [239, 205, 35],
         [240, 206, 34], [241, 207, 34], [242, 208, 35], [243, 209, 37], [244, 210, 39], [245, 211, 43], [246, 212, 47],
         [247, 214, 52], [248, 215, 57], [249, 216, 62], [250, 217, 68], [251, 218, 74], [251, 219, 80], [252, 220, 86],
         [253, 221, 93], [253, 222, 99], [254, 223, 106], [254, 224, 112], [255, 225, 119], [255, 226, 126],
         [255, 227, 132], [255, 228, 139], [255, 229, 146], [255, 230, 152], [255, 231, 159], [255, 232, 165],
         [255, 233, 171], [255, 235, 177], [255, 236, 183], [254, 237, 189], [254, 238, 195], [254, 239, 201],
         [253, 240, 206], [253, 241, 211], [253, 242, 216], [253, 243, 221], [252, 245, 226], [252, 246, 231],
         [252, 247, 235], [252, 248, 239], [252, 249, 243]],
        category="Color blind", flags=Palette.ColorBlindSafe
    ),
    'linear_tritanopic_krjcw_5_95_c24': ContinuousPalette(
        'Linear tritanopic', 'linear_tritanopic_krjcw_5_95_c24',
        [[17, 17, 17], [20, 17, 17], [22, 18, 17], [24, 18, 17], [26, 18, 17], [28, 19, 17], [30, 19, 17], [32, 19, 17],
         [34, 20, 17], [35, 20, 17], [37, 20, 17], [39, 20, 17], [41, 21, 17], [42, 21, 18], [44, 21, 18], [46, 21, 18],
         [47, 22, 18], [49, 22, 18], [51, 22, 18], [52, 22, 18], [54, 22, 19], [56, 22, 19], [57, 23, 19], [59, 23, 19],
         [61, 23, 19], [62, 23, 19], [64, 23, 20], [65, 24, 20], [67, 24, 20], [68, 24, 20], [70, 24, 21], [71, 25, 21],
         [73, 25, 21], [75, 25, 21], [76, 25, 22], [78, 26, 22], [79, 26, 22], [81, 26, 22], [82, 26, 23], [83, 27, 23],
         [85, 27, 23], [86, 27, 24], [88, 28, 24], [89, 28, 24], [91, 28, 25], [92, 29, 25], [93, 29, 25], [95, 30, 26],
         [96, 30, 26], [97, 31, 27], [99, 31, 27], [100, 32, 28], [101, 32, 28], [102, 33, 29], [104, 33, 29],
         [105, 34, 30], [106, 35, 30], [107, 35, 31], [108, 36, 31], [110, 36, 32], [111, 37, 33], [112, 38, 33],
         [113, 39, 34], [114, 39, 35], [115, 40, 35], [116, 41, 36], [117, 42, 37], [118, 43, 37], [119, 44, 38],
         [120, 45, 39], [121, 46, 40], [122, 46, 41], [122, 47, 41], [123, 48, 42], [124, 49, 43], [125, 51, 44],
         [125, 52, 45], [126, 53, 46], [127, 54, 47], [127, 55, 48], [128, 56, 49], [128, 57, 50], [129, 58, 51],
         [129, 60, 53], [130, 61, 54], [130, 62, 55], [130, 63, 56], [131, 65, 57], [131, 66, 59], [131, 67, 60],
         [131, 69, 61], [131, 70, 63], [131, 72, 64], [131, 73, 66], [131, 74, 67], [131, 76, 69], [131, 77, 70],
         [131, 79, 72], [131, 80, 73], [130, 81, 75], [130, 83, 76], [130, 84, 78], [130, 85, 79], [130, 87, 81],
         [129, 88, 82], [129, 89, 84], [129, 91, 86], [129, 92, 87], [128, 93, 89], [128, 95, 90], [128, 96, 92],
         [128, 97, 93], [127, 99, 95], [127, 100, 97], [126, 101, 98], [126, 103, 100], [126, 104, 101],
         [125, 105, 103], [125, 106, 105], [124, 108, 106], [124, 109, 108], [123, 110, 110], [123, 112, 111],
         [122, 113, 113], [122, 114, 115], [121, 115, 116], [120, 117, 118], [120, 118, 120], [119, 119, 121],
         [118, 121, 123], [118, 122, 125], [117, 123, 126], [116, 124, 128], [115, 126, 130], [114, 127, 132],
         [114, 128, 133], [113, 129, 135], [112, 131, 137], [111, 132, 139], [110, 133, 140], [109, 134, 142],
         [107, 136, 144], [106, 137, 146], [105, 138, 147], [104, 140, 149], [103, 141, 151], [101, 142, 153],
         [100, 143, 155], [98, 145, 156], [97, 146, 158], [95, 147, 160], [94, 148, 162], [92, 150, 164],
         [90, 151, 166], [88, 152, 167], [86, 153, 169], [84, 155, 171], [82, 156, 173], [80, 157, 175], [77, 158, 177],
         [75, 160, 179], [72, 161, 181], [69, 162, 182], [66, 164, 184], [63, 165, 186], [60, 166, 188], [56, 167, 190],
         [53, 168, 192], [50, 170, 193], [46, 171, 195], [43, 172, 197], [40, 173, 198], [36, 174, 200], [33, 175, 201],
         [29, 177, 203], [26, 178, 204], [23, 179, 206], [19, 180, 207], [16, 181, 208], [13, 182, 210], [11, 183, 211],
         [9, 184, 212], [8, 185, 213], [8, 186, 215], [9, 187, 216], [11, 188, 217], [13, 189, 218], [16, 190, 219],
         [19, 191, 220], [22, 192, 221], [26, 193, 222], [29, 194, 223], [33, 195, 224], [36, 196, 225], [40, 197, 226],
         [43, 198, 227], [47, 199, 227], [50, 200, 228], [54, 201, 229], [57, 202, 230], [61, 203, 230], [64, 203, 231],
         [68, 204, 232], [71, 205, 233], [75, 206, 233], [78, 207, 234], [81, 208, 234], [85, 209, 235], [88, 209, 235],
         [92, 210, 236], [95, 211, 236], [98, 212, 237], [102, 213, 237], [105, 213, 238], [108, 214, 238],
         [112, 215, 239], [115, 216, 239], [118, 217, 239], [121, 217, 240], [125, 218, 240], [128, 219, 240],
         [131, 220, 241], [135, 220, 241], [138, 221, 241], [141, 222, 241], [144, 223, 242], [148, 223, 242],
         [151, 224, 242], [154, 225, 242], [157, 225, 242], [160, 226, 242], [164, 227, 243], [167, 227, 243],
         [170, 228, 243], [173, 229, 243], [177, 229, 243], [180, 230, 243], [183, 231, 243], [186, 231, 243],
         [189, 232, 243], [193, 232, 243], [196, 233, 243], [199, 234, 243], [202, 234, 243], [205, 235, 243],
         [209, 235, 242], [212, 236, 242], [215, 236, 242], [218, 237, 242], [221, 238, 242], [225, 238, 242],
         [228, 239, 241], [231, 239, 241], [234, 240, 241], [237, 240, 241], [241, 241, 241]],
        category="Color blind", flags=Palette.ColorBlindSafe
    ),
    'isoluminant_cgo_80_c38': ContinuousPalette(
        'Isoluminant', 'isoluminant_cgo_80_c38',
        [[112, 209, 255], [112, 210, 255], [112, 210, 255], [112, 210, 255], [112, 210, 255], [112, 210, 254],
         [112, 210, 254], [112, 210, 253], [112, 210, 252], [112, 210, 251], [112, 210, 250], [112, 210, 250],
         [113, 211, 249], [113, 211, 248], [113, 211, 247], [113, 211, 247], [113, 211, 246], [113, 211, 245],
         [113, 211, 244], [113, 211, 243], [113, 211, 243], [113, 211, 242], [113, 211, 241], [114, 212, 240],
         [114, 212, 239], [114, 212, 238], [114, 212, 238], [114, 212, 237], [114, 212, 236], [114, 212, 235],
         [114, 212, 234], [115, 212, 234], [115, 212, 233], [115, 212, 232], [115, 212, 231], [115, 212, 230],
         [115, 213, 229], [115, 213, 229], [116, 213, 228], [116, 213, 227], [116, 213, 226], [116, 213, 225],
         [116, 213, 225], [116, 213, 224], [116, 213, 223], [117, 213, 222], [117, 213, 221], [117, 213, 220],
         [117, 213, 219], [117, 213, 219], [118, 213, 218], [118, 214, 217], [118, 214, 216], [118, 214, 215],
         [118, 214, 214], [119, 214, 214], [119, 214, 213], [119, 214, 212], [119, 214, 211], [119, 214, 210],
         [120, 214, 209], [120, 214, 208], [120, 214, 208], [120, 214, 207], [121, 214, 206], [121, 214, 205],
         [121, 214, 204], [122, 214, 203], [122, 214, 202], [122, 214, 201], [122, 214, 201], [123, 214, 200],
         [123, 215, 199], [123, 215, 198], [124, 215, 197], [124, 215, 196], [124, 215, 195], [125, 215, 194],
         [125, 215, 193], [125, 215, 193], [126, 215, 192], [126, 215, 191], [126, 215, 190], [127, 215, 189],
         [127, 215, 188], [128, 215, 187], [128, 215, 186], [129, 215, 185], [129, 215, 184], [129, 215, 184],
         [130, 215, 183], [130, 215, 182], [131, 215, 181], [131, 215, 180], [132, 215, 179], [132, 215, 178],
         [133, 215, 177], [133, 215, 176], [134, 215, 175], [134, 215, 174], [135, 215, 173], [136, 215, 172],
         [136, 215, 172], [137, 215, 171], [137, 215, 170], [138, 215, 169], [139, 215, 168], [139, 215, 167],
         [140, 215, 166], [141, 214, 165], [141, 214, 164], [142, 214, 163], [143, 214, 162], [144, 214, 161],
         [144, 214, 160], [145, 214, 160], [146, 214, 159], [147, 214, 158], [147, 214, 157], [148, 214, 156],
         [149, 214, 155], [150, 214, 154], [151, 213, 153], [152, 213, 153], [153, 213, 152], [154, 213, 151],
         [154, 213, 150], [155, 213, 149], [156, 213, 148], [157, 213, 148], [158, 212, 147], [159, 212, 146],
         [160, 212, 145], [161, 212, 144], [162, 212, 144], [163, 212, 143], [164, 211, 142], [165, 211, 142],
         [166, 211, 141], [167, 211, 140], [168, 211, 140], [169, 211, 139], [170, 210, 138], [171, 210, 138],
         [172, 210, 137], [173, 210, 136], [174, 210, 136], [175, 209, 135], [176, 209, 135], [177, 209, 134],
         [178, 209, 134], [179, 209, 133], [180, 208, 133], [181, 208, 132], [182, 208, 132], [183, 208, 131],
         [184, 207, 131], [185, 207, 130], [186, 207, 130], [187, 207, 129], [188, 207, 129], [189, 206, 128],
         [190, 206, 128], [191, 206, 127], [192, 206, 127], [193, 205, 127], [194, 205, 126], [195, 205, 126],
         [196, 205, 125], [197, 204, 125], [197, 204, 125], [198, 204, 124], [199, 204, 124], [200, 203, 124],
         [201, 203, 123], [202, 203, 123], [203, 203, 123], [204, 202, 122], [205, 202, 122], [206, 202, 122],
         [207, 202, 121], [208, 201, 121], [209, 201, 121], [209, 201, 121], [210, 201, 120], [211, 200, 120],
         [212, 200, 120], [213, 200, 120], [214, 199, 119], [215, 199, 119], [216, 199, 119], [217, 199, 119],
         [217, 198, 119], [218, 198, 119], [219, 198, 118], [220, 197, 118], [221, 197, 118], [222, 197, 118],
         [223, 197, 118], [224, 196, 118], [224, 196, 118], [225, 196, 118], [226, 195, 118], [227, 195, 118],
         [228, 195, 118], [229, 194, 118], [229, 194, 118], [230, 194, 118], [231, 194, 118], [232, 193, 118],
         [233, 193, 118], [233, 193, 118], [234, 192, 118], [235, 192, 118], [236, 192, 118], [237, 191, 118],
         [237, 191, 118], [238, 191, 118], [239, 190, 118], [240, 190, 119], [241, 190, 119], [241, 189, 119],
         [242, 189, 119], [243, 189, 119], [244, 188, 119], [244, 188, 120], [245, 188, 120], [246, 188, 120],
         [247, 187, 120], [247, 187, 121], [248, 187, 121], [249, 186, 121], [249, 186, 121], [250, 186, 122],
         [251, 185, 122], [252, 185, 122], [252, 185, 122], [253, 184, 123], [254, 184, 123], [254, 184, 123],
         [255, 183, 124], [255, 183, 124], [255, 183, 124], [255, 182, 125], [255, 182, 125], [255, 182, 125],
         [255, 181, 126], [255, 181, 126], [255, 181, 127], [255, 180, 127], [255, 180, 127], [255, 180, 128],
         [255, 179, 128], [255, 179, 129], [255, 179, 129], [255, 178, 129]],
        category="Other"
    ),
    'rainbow_bgyr_35_85_c73': ContinuousPalette(
        'Rainbow', 'rainbow_bgyr_35_85_c73',
        [[0, 53, 249], [0, 56, 246], [0, 58, 243], [0, 61, 240], [0, 63, 237], [0, 66, 234], [0, 68, 231], [0, 71, 228],
         [0, 73, 225], [0, 75, 223], [0, 77, 220], [0, 79, 217], [0, 81, 214], [0, 83, 211], [0, 85, 208], [0, 87, 205],
         [0, 89, 202], [0, 91, 199], [0, 92, 196], [0, 94, 194], [0, 96, 191], [0, 98, 188], [0, 99, 185],
         [0, 101, 182], [0, 103, 179], [0, 104, 176], [0, 106, 174], [0, 108, 171], [0, 109, 168], [0, 111, 165],
         [0, 112, 163], [0, 113, 160], [0, 115, 157], [0, 116, 155], [0, 117, 152], [0, 118, 150], [7, 119, 147],
         [14, 120, 145], [20, 122, 142], [24, 123, 140], [28, 124, 137], [32, 125, 135], [35, 126, 133], [38, 127, 130],
         [41, 128, 128], [43, 129, 126], [45, 130, 123], [47, 131, 121], [49, 132, 118], [51, 133, 116], [52, 134, 114],
         [53, 135, 111], [55, 136, 109], [56, 137, 106], [57, 138, 104], [58, 139, 101], [59, 140, 99], [59, 141, 96],
         [60, 142, 94], [61, 143, 91], [61, 144, 88], [62, 145, 86], [62, 146, 83], [62, 147, 80], [63, 148, 78],
         [63, 149, 75], [63, 150, 72], [63, 152, 69], [63, 153, 66], [63, 154, 63], [63, 155, 60], [63, 156, 57],
         [63, 157, 53], [63, 158, 50], [63, 159, 47], [63, 160, 43], [63, 161, 40], [64, 162, 36], [64, 163, 33],
         [65, 164, 30], [66, 165, 27], [68, 166, 24], [70, 166, 21], [72, 167, 19], [74, 168, 17], [77, 169, 16],
         [79, 169, 15], [82, 170, 14], [85, 171, 13], [88, 171, 13], [90, 172, 13], [93, 172, 14], [96, 173, 14],
         [99, 174, 14], [101, 174, 14], [104, 175, 15], [106, 175, 15], [109, 176, 16], [112, 177, 16], [114, 177, 16],
         [117, 178, 17], [119, 178, 17], [122, 179, 17], [124, 180, 18], [126, 180, 18], [129, 181, 19], [131, 181, 19],
         [134, 182, 19], [136, 182, 20], [138, 183, 20], [141, 183, 20], [143, 184, 21], [145, 185, 21], [148, 185, 21],
         [150, 186, 22], [152, 186, 22], [154, 187, 23], [157, 187, 23], [159, 188, 23], [161, 188, 24], [163, 189, 24],
         [166, 189, 24], [168, 190, 25], [170, 191, 25], [172, 191, 26], [175, 192, 26], [177, 192, 26], [179, 193, 27],
         [181, 193, 27], [183, 194, 27], [186, 194, 28], [188, 195, 28], [190, 195, 28], [192, 196, 29], [194, 196, 29],
         [196, 197, 30], [199, 197, 30], [201, 198, 30], [203, 198, 31], [205, 199, 31], [207, 199, 31], [209, 200, 32],
         [211, 200, 32], [214, 201, 33], [216, 201, 33], [218, 202, 33], [220, 202, 34], [222, 203, 34], [224, 203, 34],
         [226, 203, 35], [229, 204, 35], [231, 204, 35], [233, 205, 36], [235, 205, 36], [237, 205, 36], [239, 205, 37],
         [241, 205, 37], [242, 205, 37], [244, 205, 37], [245, 205, 37], [247, 204, 37], [248, 204, 36], [249, 203, 36],
         [250, 202, 36], [251, 201, 35], [251, 200, 35], [252, 199, 34], [252, 197, 34], [253, 196, 33], [253, 195, 33],
         [253, 193, 32], [253, 192, 32], [254, 191, 31], [254, 189, 30], [254, 188, 30], [254, 187, 29], [254, 185, 29],
         [255, 184, 28], [255, 182, 27], [255, 181, 27], [255, 180, 26], [255, 178, 25], [255, 177, 25], [255, 176, 24],
         [255, 174, 24], [255, 173, 23], [255, 171, 22], [255, 170, 22], [255, 168, 21], [255, 167, 20], [255, 166, 20],
         [255, 164, 19], [255, 163, 18], [255, 161, 18], [255, 160, 17], [255, 158, 16], [255, 157, 16], [255, 156, 15],
         [255, 154, 14], [255, 153, 13], [255, 151, 13], [255, 150, 12], [255, 148, 11], [255, 147, 10], [255, 145, 10],
         [255, 144, 9], [255, 142, 8], [255, 141, 7], [255, 139, 7], [255, 138, 6], [255, 136, 5], [255, 134, 5],
         [255, 133, 4], [255, 131, 3], [255, 130, 3], [255, 128, 2], [255, 127, 2], [255, 125, 1], [255, 123, 1],
         [255, 122, 0], [255, 120, 0], [255, 118, 0], [255, 117, 0], [255, 115, 0], [255, 113, 0], [255, 112, 0],
         [255, 110, 0], [255, 108, 0], [255, 106, 0], [255, 104, 0], [255, 103, 0], [255, 101, 0], [255, 99, 0],
         [255, 97, 0], [255, 95, 0], [255, 93, 0], [255, 91, 0], [255, 89, 0], [255, 87, 0], [255, 85, 0], [255, 83, 0],
         [255, 81, 0], [255, 79, 0], [255, 76, 0], [255, 74, 0], [255, 72, 0], [255, 69, 0], [255, 67, 0], [255, 64, 0],
         [255, 61, 0], [255, 59, 0], [255, 56, 0], [255, 53, 0], [255, 49, 0], [255, 46, 0], [255, 42, 0]],
        category="Other"
    ),
}

DefaultContinuousPaletteName = "linear_bgy_10_95_c74"
DefaultContinuousPalette = ContinuousPalettes[DefaultContinuousPaletteName]


class ColorIcon(QIcon):
    def __init__(self, color, size=12):
        p = QPixmap(size, size)
        p.fill(color)
        super().__init__(p)


def get_default_curve_colors(n):
    if n <= len(Dark2Colors):
        return list(Dark2Colors)[:n]
    if n <= len(DefaultRGBColors):
        return list(DefaultRGBColors)[:n]
    else:
        return list(LimitedDiscretePalette(n))


def patch_variable_colors():
    # This function patches Variable with properties and private attributes:
    # pylint: disable=protected-access
    from Orange.data import Variable, DiscreteVariable, ContinuousVariable

    def get_colors(var):
        return var._colors

    def set_colors(var, colors):
        var._colors = colors
        var._palette = None
        var.attributes["colors"] = [
            color_to_hex(color) if isinstance(color, (Sequence, np.ndarray))
            else color
            for color in colors]
        if "palette" in var.attributes:
            del var.attributes["palette"]

    def get_palette(var):
        return var._palette

    def set_palette(var, palette):
        var._palette = palette
        var.attributes["palette"] = palette.name
        var._colors = None
        if "colors" in var.attributes:
            del var.attributes["colors"]

    def continuous_get_colors(var):
        warnings.warn("ContinuousVariable.color is deprecated; "
                      "use ContinuousVariable.palette",
                      DeprecationWarning, stacklevel=2)
        if var._colors is None:
            try:
                col1, col2, black = var.attributes["colors"]
                var._colors = (hex_to_color(col1), hex_to_color(col2), black)
            except (KeyError, ValueError):  # unavailable or invalid
                if var._palette or "palette" in var.attributes:
                    palette = var.palette
                    col1 = tuple(palette.palette[0])
                    col2 = tuple(palette.palette[-1])
                    black = bool(palette.flags & palette.Diverging)
                    var._colors = col1, col2, black
                else:
                    var._colors = ((0, 0, 255), (255, 255, 0), False)
        return var._colors

    def continuous_get_palette(var):
        if var._palette is None:
            if "palette" in var.attributes:
                var._palette = ContinuousPalettes.get(var.attributes["palette"],
                                                      DefaultContinuousPalette)
            elif var._colors is not None or "colors" in var.attributes:
                col1, col2, black = var.colors
                var._palette = ContinuousPalette.from_colors(col1, col2, black)
            else:
                var._palette = DefaultContinuousPalette
        return var._palette

    def discrete_get_colors(var):
        if var._colors is None or len(var._colors) < len(var.values):
            if var._palette is not None or "palette" in var.attributes:
                var._colors = var.palette.palette[:len(var.values)]
            else:
                var._colors = np.empty((0, 3), dtype=object)
            colors = var.attributes.get("colors")
            if colors:
                try:
                    var._colors = np.vstack(
                        ([hex_to_color(color) for color in colors],
                         var._colors[len(colors):]))
                except ValueError:
                    pass
            if len(var._colors) < len(var.values):
                var._colors = LimitedDiscretePalette(len(var.values)).palette
            var._colors.flags.writeable = False
        return var._colors

    def discrete_set_colors(var, colors):
        colors = colors.copy()
        colors.flags.writeable = False
        set_colors(var, colors)

    def discrete_get_palette(var):
        if var._palette is None:
            if "palette" in var.attributes:
                var._palette = DiscretePalettes.get(var.attributes["palette"],
                                                    DefaultDiscretePalette)
            elif var._colors is not None or "colors" in var.attributes:
                var._palette = DiscretePalette.from_colors(var.colors)
            else:
                var._palette = LimitedDiscretePalette(len(var.values))
        return var._palette

    Variable._colors = None
    Variable._palette = None
    Variable.colors = property(get_colors, set_colors)
    Variable.palette = property(get_palette, set_palette)

    DiscreteVariable.colors = property(discrete_get_colors, discrete_set_colors)
    DiscreteVariable.palette = property(discrete_get_palette, set_palette)

    ContinuousVariable.colors = property(continuous_get_colors, set_colors)
    ContinuousVariable.palette = property(continuous_get_palette, set_palette)
