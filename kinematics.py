from collections import defaultdict
import sys
import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mplstereonet

PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI
STRIKE_TOLERANCE = 20.0
DAYLIGHT_TOLERANCE = 90.0


class KinematicAnalyzer:
    """
    Class for kinematic slope stability analysis.
    Handles data loading, validation, analysis, and plotting.
    """

    def __init__(self, data=None, file_path=None):
        self.df = None
        self.results = None
        self.slope_dip_dir = None
        self.slope_dip = None
        self.friction = None
        if file_path:
            self.load_data(file_path)
        elif data is not None:
            self.df = data.copy()
            self._validate_data()

    def load_data(self, file_path):
        """Helper: Load CSV and validate."""
        try:
            self.df = pd.read_csv(file_path)
            self._validate_data()
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")

    def _validate_data(self):
        """Helper: Check columns and angle bounds."""
        required_cols = ['Dip_Direction', 'Dip_Angle']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"CSV must have columns: {required_cols}")
        if not (self.df['Dip_Direction'].between(0, 360).all() and
                self.df['Dip_Angle'].between(0, 90).all()):
            raise ValueError(
                "Angles must be: Dip_Direction 0-360°, Dip_Angle 0-90°")

    def set_parameters(self, slope_dip_dir, slope_dip, friction):
        """Helper: Set and validate inputs."""
        self.slope_dip_dir = self._clamp_angle(slope_dip_dir)
        self.slope_dip = self._clamp_angle(slope_dip, max_val=90)
        self.friction = self._clamp_angle(friction, max_val=45)

    def _clamp_angle(self, val, max_val=360):
        """Helper: Clamp angle to 0-max_val."""
        return max(0, min(max_val, val))

    # Core functions (unchanged, but as methods)
    def azimuth_diff(self, a1, a2):
        diff = abs(a1 - a2) % 360
        return min(diff, 360 - diff)

    def plane_to_normal(self, dip_dir, dip):
        pole_plunge = 90.0 - dip
        theta = (90.0 - pole_plunge) * DEG_TO_RAD
        phi = dip_dir * DEG_TO_RAD
        nx = math.sin(theta) * math.cos(phi)
        ny = math.sin(theta) * math.sin(phi)
        nz = math.cos(theta)
        norm = math.sqrt(nx**2 + ny**2 + nz**2)
        return np.array([nx, ny, nz]) / norm

    def line_from_vector(self, vec):
        vx, vy, vz = vec
        horiz = math.sqrt(vx**2 + vy**2)
        if horiz == 0:
            return 0.0, 90.0 if vz < 0 else 0.0
        plunge = math.degrees(math.atan2(-vz, horiz))
        trend = math.degrees(math.atan2(vy, vx)) % 360
        return trend, plunge

    def compute_intersection(self, plane1, plane2):
        n1 = self.plane_to_normal(plane1['dip_dir'], plane1['dip'])
        n2 = self.plane_to_normal(plane2['dip_dir'], plane2['dip'])
        inter_vec = np.cross(n1, n2)
        if np.linalg.norm(inter_vec) == 0:
            return None, None
        trend, plunge = self.line_from_vector(inter_vec)
        if plunge > 90:
            trend = (trend + 180) % 360
            plunge = 180 - plunge
        return trend, plunge

    def check_planar_failure(self, plane):
        dip_dir, dip = plane['dip_dir'], plane['dip']
        if self.azimuth_diff(dip_dir, self.slope_dip_dir) > STRIKE_TOLERANCE:
            return False, 0.0  # FS placeholder
        if dip >= self.slope_dip:
            return False, 0.0
        if dip <= self.friction:
            return False, 0.0
        # Simple FS for planar: tan(phi)/tan(dip)
        fs = math.tan(math.radians(self.friction)) / \
            math.tan(math.radians(dip))
        return fs < 1.0, fs

    def check_wedge_failure(self, inter_trend, inter_plunge):
        if inter_plunge >= self.slope_dip:
            return False, 0.0
        if self.azimuth_diff(inter_trend, self.slope_dip_dir) > DAYLIGHT_TOLERANCE:
            return False, 0.0
        if inter_plunge <= self.friction:
            return False, 0.0
        # Approx FS for wedge (simplified; use full LEM for accuracy)
        fs = math.tan(math.radians(self.friction)) / \
            math.tan(math.radians(inter_plunge))
        return fs < 1.0, fs

    def analyze(self):
        """Main analysis method."""
        if self.df is None:
            raise ValueError("Load data first.")
        df_processed = self.df.copy()
        df_processed.columns = ['dip_dir', 'dip']
        total_planes = len(df_processed)

        # Planar
        planar_potential = []
        for idx, row in df_processed.iterrows():
            is_failure, fs = self.check_planar_failure(row)
            if is_failure:
                planar_potential.append((idx, fs))

        # Wedge
        wedge_potential = []
        for idx1, idx2 in combinations(range(total_planes), 2):
            plane1 = df_processed.iloc[idx1].to_dict()
            plane2 = df_processed.iloc[idx2].to_dict()
            inter_trend, inter_plunge = self.compute_intersection(
                plane1, plane2)
            if inter_trend is None:
                continue
            is_failure, fs = self.check_wedge_failure(
                inter_trend, inter_plunge)
            if is_failure:
                wedge_potential.append(
                    (idx1, idx2, inter_trend, inter_plunge, fs))

        self.results = {
            'planar_potential': planar_potential,  # Now (idx, fs)
            # Now (idx1, idx2, trend, plunge, fs)
            'wedge_potential': wedge_potential,
            'counts': {'planar': len(planar_potential), 'wedge': len(wedge_potential), 'total_planes': total_planes},
            'df_processed': df_processed
        }
        return self.results

    def get_summary(self):
        """Helper: Generate text summary."""
        if not self.results:
            return "Run analysis first."
        r = self.results
        summary = f"=== SUMMARY ===\nTotal Planes: {r['counts']['total_planes']}\n"
        summary += f"Potential Planar (FS<1): {r['counts']['planar']}\n"
        if r['counts']['planar'] > 0:
            summary += f"Planar: {[(i, f'FS={fs:.2f}')
                                   for i, fs in r['planar_potential']]}\n"
        summary += f"Potential Wedge (FS<1): {r['counts']['wedge']}\n"
        if r['counts']['wedge'] > 0:
            summary += f"Wedges: {[(i1, i2, f'FS={fs:.2f}')
                                   for i1, i2, _, _, fs in r['wedge_potential']]}\n"
        if r['counts']['planar'] + r['counts']['wedge'] == 0:
            summary += "Stable geometrically."
        else:
            summary += "WARNING: Proceed to detailed FS."
        return summary

    def save_results(self, file_path):
        """Helper: Save to CSV with FS."""
        if not self.results:
            raise ValueError("Run analysis first.")
        r = self.results
        planar_df = pd.DataFrame(r['planar_potential'], columns=[
                                 'Plane_Index', 'FS'])
        planar_df['Type'] = 'Planar'
        wedge_df = pd.DataFrame(r['wedge_potential'], columns=[
                                'Plane1_Index', 'Plane2_Index', 'Inter_Trend', 'Inter_Plunge', 'FS'])
        wedge_df['Type'] = 'Wedge'
        combined = pd.concat([planar_df, wedge_df[['Plane1_Index', 'Plane2_Index', 'FS', 'Type']].rename(
            columns={'Plane1_Index': 'Plane_Index', 'Plane2_Index': 'N/A'})], ignore_index=True)
        combined.to_csv(file_path, index=False)

    def plot_stereonet(self, projection='equal_area', save_path=None):
        """Plot and optionally save; returns fig with metadata for hover."""
        if not self.results:
            raise ValueError("Run analysis first.")
        r = self.results
        proj_param = 'stereonet' if projection == 'equal_area' else 'wulff'

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=proj_param)
        ax.grid(True)

        # Store metadata for hover: dict of (x,y) -> info
        hover_data = defaultdict(list)  # (az_rad, r) -> list of dicts

        # Plot poles with hover data
        first_planar = True
        for idx, fs in r['planar_potential']:
            row = r['df_processed'].iloc[idx]
            pole_az = (row['dip_dir'] + 180) % 360
            pole_plunge = 90 - row['dip']
            x = math.radians(pole_az)
            if projection == 'equal_area':
                r_val = 2 * math.asin(math.sin(math.radians(pole_plunge) / 2))
            else:
                r_val = math.radians(pole_plunge)
            artist = ax.pole(pole_az, pole_plunge, color='red', marker='o', markersize=8,
                             label='Potential Planar' if first_planar else None)
            hover_data[(x, r_val)].append({
                'type': 'pole', 'index': idx, 'dip_dir': row['dip_dir'], 'dip': row['dip'],
                'fs': fs, 'color': 'red'
            })
            first_planar = False

        for idx, row in r['df_processed'].iterrows():
            if idx in [i for i, _ in r['planar_potential']]:
                continue
            pole_az = (row['dip_dir'] + 180) % 360
            pole_plunge = 90 - row['dip']
            x = math.radians(pole_az)
            if projection == 'equal_area':
                r_val = 2 * math.asin(math.sin(math.radians(pole_plunge) / 2))
            else:
                r_val = math.radians(pole_plunge)
            ax.pole(pole_az, pole_plunge, color='blue',
                    marker='o', markersize=8)
            hover_data[(x, r_val)].append({
                'type': 'pole', 'index': idx, 'dip_dir': row['dip_dir'], 'dip': row['dip'],
                'fs': None, 'color': 'blue'
            })

        # Slope normal
        slope_pole_az = (self.slope_dip_dir + 180) % 360
        slope_pole_plunge = 90 - self.slope_dip
        x_slope = math.radians(slope_pole_az)
        if projection == 'equal_area':
            r_slope = 2 * \
                math.asin(math.sin(math.radians(slope_pole_plunge) / 2))
        else:
            r_slope = math.radians(slope_pole_plunge)
        ax.pole(slope_pole_az, slope_pole_plunge, marker='s',
                color='green', markersize=12, label='Slope Normal')
        hover_data[(x_slope, r_slope)].append({
            'type': 'slope_normal', 'dir': self.slope_dip_dir, 'dip': self.slope_dip
        })

        # Slope great circle (approximate center for hover; line is harder, so sample points)
        ax.plane(self.slope_dip_dir, self.slope_dip, 'g--',
                 linewidth=2, label='Slope Great Circle')
        # Add hover for a midpoint on great circle (e.g., 180° from normal)
        gc_x = math.radians((self.slope_dip_dir + 180) % 360)
        gc_r = math.pi / 2  # Equatorial radius
        hover_data[(gc_x, gc_r)].append({
            'type': 'great_circle', 'dir': self.slope_dip_dir, 'dip': self.slope_dip
        })

        # Wedges (lines from (0,0) to end point)
        if r['counts']['wedge'] > 0:
            first_wedge = True
            for wp_idx, (idx1, idx2, inter_trend, inter_plunge, fs) in enumerate(r['wedge_potential']):
                end_x = math.radians(inter_trend)
                if projection == 'equal_area':
                    end_r = 2 * \
                        math.asin(math.sin(math.radians(inter_plunge) / 2))
                else:
                    end_r = math.radians(inter_plunge)
                ax.line(inter_trend, inter_plunge, color='orange', linewidth=2,
                        label='Potential Wedge' if first_wedge else None)
                # Store at endpoint for hover
                hover_data[(end_x, end_r)].append({
                    'type': 'wedge', 'planes': (idx1, idx2), 'trend': inter_trend, 'plunge': inter_plunge, 'fs': fs
                })
                first_wedge = False

        ax.legend(loc='upper right')
        title = f'Stereonet ({projection}) | Planar: {r["counts"]["planar"]} | Wedge: {r["counts"]["wedge"]}'
        ax.set_title(title)

        # Attach hover data to fig for use in GUI
        fig.hover_data = hover_data
        fig.hover_tolerance = 0.05  # Angular tolerance for hover detection

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
