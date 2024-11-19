from copy import deepcopy
from typing import Union

import numpy as np
from bokeh.models import Arrow, NormalHead, ColumnDataSource
from bokeh.palettes import Category10_10
from bokeh.plotting import figure
from bokeh.io import export_svg
from bokeh.transform import factor_cmap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .subplot import Figure


class PCAPlot(Figure):
    def __init__(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        x_axis: str,
        y_axis: str,
    ) -> None:
        super().__init__()
        self.train_features = train_features
        self.test_features = test_features
        self.x_axis = x_axis
        self.y_axis = y_axis

        self.active_index = None
        self.modified_pca = None
        self.modified_x = None
        self.modified_y = None

        combined_train_test_features = np.vstack(
            [self.train_features, self.test_features]
        )
        self.scaler = StandardScaler()
        self.train_features = self.scaler.fit_transform(self.train_features)
        self.pca = PCA(n_components=2).fit(self.train_features)
        self.train_pca_data = self.pca.transform(self.train_features)

        self.test_features = self.scaler.transform(self.test_features)
        self.test_pca_data = self.pca.transform(self.test_features)

        # Combine train and test data
        combined_pca_data = np.vstack([self.train_pca_data, self.test_pca_data])
        index_array = np.concatenate(
            [np.arange(len(self.train_features)), np.arange(len(self.test_features))]
        )

        self.labels = ["train data"] * len(self.train_features) + ["test data"] * len(
            self.test_features
        )

        # Build source data based on input axes
        self.data_dict = {
            "trend_strength": combined_train_test_features[:, 0],
            "trend_slope": combined_train_test_features[:, 1],
            "trend_linearity": combined_train_test_features[:, 2],
            "seasonal_strength": combined_train_test_features[:, 3],
            "pca_1": combined_pca_data[:, 0],
            "pca_2": combined_pca_data[:, 1],
        }

        self.map_from_axes_to_index = {
            "trend_strength": 0,
            "trend_slope": 1,
            "trend_linearity": 2,
            "seasonal_strength": 3,
            "pca_1": 4,
            "pca_2": 5,
        }

        x_data = self.data_dict[x_axis]
        y_data = self.data_dict[y_axis]

        self.source = ColumnDataSource(
            data={
                "x": x_data,
                "y": y_data,
                "ts_index": index_array,
                "label": self.labels,
            }
        )
        self.orig_point_source = ColumnDataSource(data={"x": [np.nan], "y": [np.nan]})
        self.modified_point_source = ColumnDataSource(
            data={"x": [np.nan], "y": [np.nan]}
        )
        self.arrow_source = ColumnDataSource(
            data={
                "x_start": [np.nan],
                "y_start": [np.nan],
                "x_end": [np.nan],
                "y_end": [np.nan],
            }
        )

        tooltips = [
            ("index", "@label at index @ts_index"),
            ("x val", "@x"),
            ("y val", "@y"),
        ]

        self.fig = figure(
            x_axis_label=x_axis,
            y_axis_label=y_axis,
            tools="pan, box_zoom, wheel_zoom, reset, tap",
            tooltips=tooltips,
            height=400,
            width=800,
        )

        self.fig.circle(
            "x",
            "y",
            source=self.source,
            selection_color=Category10_10[2],
            nonselection_alpha=0.1,
            color=factor_cmap("label", Category10_10, ["train data", "test data"]),
            legend_field="label",
        )

        # Invisible circle to set legend for the selected point
        self.fig.circle(
            "x",
            "y",
            source=self.orig_point_source,
            color=Category10_10[2],
            legend_label="original position",
        )
        self.fig.circle(
            "x",
            "y",
            source=self.modified_point_source,
            color=Category10_10[3],
            legend_label="modified position",
        )

        self.fig.add_layout(
            Arrow(
                end=NormalHead(fill_color="red"),
                x_start="x_start",
                y_start="y_start",
                x_end="x_end",
                y_end="y_end",
                source=self.arrow_source,
                line_width=2,
                line_color="red",
            )
        )

        self.fig.legend.background_fill_alpha = 0
        self.fig.legend.border_line_alpha = 0

    def update_source(self) -> None:
        # Update the data in the source based on the current axes
        x_data = self.data_dict[self.x_axis]
        y_data = self.data_dict[self.y_axis]

        source_array = deepcopy(np.column_stack([x_data, y_data]))
        index_array = np.concatenate(
            [np.arange(len(self.train_features)), np.arange(len(self.test_features))]
        )

        orig_pos = [np.nan, np.nan]
        if (
            self.active_index is not None
            and self.modified_x is not None
            and self.modified_y is not None
        ):
            orig_pos = deepcopy(
                source_array[self.active_index]
            )  # Save original PCA position
            source_array[self.active_index] = (
                self.modified_x,
                self.modified_y,
            )  # Apply the modification

        # Update the main source with the transformed data
        self.source.data = dict(
            x=source_array[:, 0],  # x axis
            y=source_array[:, 1],  # y axis
            ts_index=index_array,  # Time-series indices
            label=self.labels,  # Labels for train/test data
        )
        self.orig_point_source.data = dict(x=[orig_pos[0]], y=[orig_pos[1]])
        self.modified_point_source.data = dict(x=[self.modified_x], y=[self.modified_y])
        self.modify_arrow()

    def update_axes(self, x_axis: str, y_axis: str) -> None:
        # Update axes and regenerate the plot source
        if x_axis not in self.data_dict or y_axis not in self.data_dict:
            raise ValueError(
                f"Invalid axis selection: x_axis={x_axis}, y_axis={y_axis}"
            )

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.update_source()

        self.fig.xaxis.axis_label = x_axis
        self.fig.yaxis.axis_label = y_axis

    def update_features(self, features: Union[np.ndarray, None]) -> None:
        if features is None:
            return

        self.x_axis_index = self.map_from_axes_to_index[self.x_axis]
        self.y_axis_index = self.map_from_axes_to_index[self.y_axis]

        if self.x_axis_index < 4:
            self.modified_x = features[0][self.map_from_axes_to_index[self.x_axis]]
        if self.y_axis_index < 4:
            self.modified_y = features[0][self.map_from_axes_to_index[self.y_axis]]
        scaled_features = self.scaler.transform(features)
        pca_data = self.pca.transform(scaled_features)

        if self.x_axis_index == 4:
            self.modified_x = pca_data[0, 0]
        elif self.x_axis_index == 5:
            self.modified_x = pca_data[0, 1]
        if self.y_axis_index == 4:
            self.modified_y = pca_data[0, 0]
        elif self.y_axis_index == 5:
            self.modified_y = pca_data[0, 1]
        self.modify_arrow()

    def modify_arrow(self) -> None:
        # Update the arrow to point from the original to the modified position
        if self.active_index is not None and self.modified_x is not None:
            x_start, y_start = (
                self.orig_point_source.data["x"][0],
                self.orig_point_source.data["y"][0],
            )
            x_end = self.modified_x
            y_end = self.modified_y
            self.arrow_source.data = dict(
                x_start=[x_start], y_start=[y_start], x_end=[x_end], y_end=[y_end]
            )
        else:
            self.arrow_source.data = dict(
                x_start=[np.nan], y_start=[np.nan], x_end=[np.nan], y_end=[np.nan]
            )

    def reset(self) -> None:
        if self.active_index is None:
            return

        self.modified_x = None
        self.modified_y = None
        self.modified_pca = None
        self.modify_arrow()

    def save_figure(self, path: str) -> None:
        export_svg(self.fig, filename=path)
