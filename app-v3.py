import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(layout="wide", page_title="Acoustic Phased Array Simulator")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem !important;
    }
    .flush-grid-row {
        display: flex !important;
        flex-direction: row !important;
        width: 100% !important;
        justify-content: center;
    }
    .flush-grid-row > div {
        flex: 1 1 0% !important;
        padding: 0px !important;
        margin: 0px !important;
    }
    div.stButton > button[key^="btn_"] {
        aspect-ratio: 1 / 1 !important;
        width: 100% !important;
        padding: 0px !important;
        margin: 0px !important;
        border-radius: 0px !important;
        border: 1px solid #1e1e1e !important;
        font-size: 10px !important;
        line-height: 1 !important;
    }
    .compact-grid-container {
        max-width: 220px !important;
        margin: 0 auto !important;
        border: 2px solid #333;
        border-radius: 4px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 Interactive Acoustic Phased Array Simulator")


# =============================================================================
# HELPERS
# =============================================================================
def init_state() -> None:
    if "anim_running" not in st.session_state:
        st.session_state.anim_running = False
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    if "grid_state" not in st.session_state:
        st.session_state.grid_state = np.zeros((9, 9), dtype=bool)
    if "last_scan_x_deg" not in st.session_state:
        st.session_state.last_scan_x_deg = 0.0
    if "last_scan_y_deg" not in st.session_state:
        st.session_state.last_scan_y_deg = 0.0
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = "0.1 kHz"
    if "taper" not in st.session_state:
        st.session_state.taper = "Uniform"
    if "array_mode" not in st.session_state:
        st.session_state.array_mode = "Custom grid linear sweep"
    if "orbit_trace_x" not in st.session_state:
        st.session_state.orbit_trace_x = []
    if "orbit_trace_y" not in st.session_state:
        st.session_state.orbit_trace_y = []
    if "peak3d_trace_x" not in st.session_state:
        st.session_state.peak3d_trace_x = []
    if "peak3d_trace_y" not in st.session_state:
        st.session_state.peak3d_trace_y = []
    if "peak3d_trace_z" not in st.session_state:
        st.session_state.peak3d_trace_z = []


def array_weights(coords: list[tuple[int, int]], taper: str) -> np.ndarray:
    if len(coords) == 0:
        return np.array([])

    x = np.array([c[0] for c in coords], dtype=float)
    y = np.array([c[1] for c in coords], dtype=float)
    r = np.sqrt(x**2 + y**2)

    if taper == "Uniform":
        return np.ones(len(coords), dtype=float)

    if np.max(r) == 0:
        return np.ones(len(coords), dtype=float)

    r_norm = r / np.max(r)
    if taper == "Hann-like":
        return 0.5 * (1.0 + np.cos(np.pi * r_norm))
    if taper == "Strong edge taper":
        return np.power(0.5 * (1.0 + np.cos(np.pi * r_norm)), 2)
    if taper == "Gaussian radial":
        sigma = 0.28
        return np.exp(-0.5 * (r_norm / sigma) ** 2)
    if taper == "Gaussian radial strong":
        sigma = 0.20
        return np.exp(-0.5 * (r_norm / sigma) ** 2)

    return np.ones(len(coords), dtype=float)


def compute_array_factor(
    active_elements: list[tuple[int, int]],
    d_spacing: float,
    wavelength_m: float,
    steer_x_deg: float,
    steer_y_deg: float,
    azimuth_deg: np.ndarray,
    elevation_deg,
    taper: str = "Uniform",
):
    if len(active_elements) == 0:
        if np.isscalar(elevation_deg):
            return np.zeros_like(azimuth_deg, dtype=float)
        az = np.asarray(azimuth_deg)
        el = np.asarray(elevation_deg)
        return np.zeros((len(el), len(az)), dtype=float)

    k = 2.0 * np.pi / wavelength_m
    theta_x_rad = np.radians(steer_x_deg)
    theta_y_rad = np.radians(steer_y_deg)
    weights = array_weights(active_elements, taper)

    az = np.radians(azimuth_deg)

    if np.isscalar(elevation_deg):
        total = np.zeros_like(az, dtype=complex)
        for (x, y), w in zip(active_elements, weights):
            pos_x = x * d_spacing
            pos_y = y * d_spacing
            phase_steer = k * (pos_x * np.sin(theta_x_rad) + pos_y * np.sin(theta_y_rad))
            path_diff = pos_x * np.sin(az)
            total += w * np.exp(1j * (k * path_diff - phase_steer))
        amp = np.abs(total)
        if np.max(amp) > 0:
            amp = amp / np.max(amp)
        return amp

    el = np.radians(elevation_deg)
    az_mesh, el_mesh = np.meshgrid(az, el)
    total = np.zeros_like(az_mesh, dtype=complex)
    for (x, y), w in zip(active_elements, weights):
        pos_x = x * d_spacing
        pos_y = y * d_spacing
        phase_steer = k * (pos_x * np.sin(theta_x_rad) + pos_y * np.sin(theta_y_rad))
        path_diff = pos_x * np.sin(az_mesh) * np.cos(el_mesh) + pos_y * np.sin(el_mesh)
        total += w * np.exp(1j * (k * path_diff - phase_steer))

    amp = np.abs(total)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return amp


def compute_orbit_map(
    active_elements: list[tuple[int, int]],
    d_spacing: float,
    wavelength_m: float,
    steer_x_deg: float,
    steer_y_deg: float,
    scan_x_deg: np.ndarray,
    scan_y_deg: np.ndarray,
    taper: str,
):
    if len(active_elements) == 0:
        return np.zeros((len(scan_y_deg), len(scan_x_deg)), dtype=float)

    k = 2.0 * np.pi / wavelength_m
    steer_x_rad = np.radians(steer_x_deg)
    steer_y_rad = np.radians(steer_y_deg)
    sx = np.radians(scan_x_deg)
    sy = np.radians(scan_y_deg)
    sx_mesh, sy_mesh = np.meshgrid(sx, sy)
    weights = array_weights(active_elements, taper)

    total = np.zeros_like(sx_mesh, dtype=complex)
    for (x, y), w in zip(active_elements, weights):
        pos_x = x * d_spacing
        pos_y = y * d_spacing
        phase_steer = k * (pos_x * np.sin(steer_x_rad) + pos_y * np.sin(steer_y_rad))
        scan_phase = k * (pos_x * np.sin(sx_mesh) + pos_y * np.sin(sy_mesh))
        total += w * np.exp(1j * (scan_phase - phase_steer))

    amp = np.abs(total)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return amp


def estimate_metrics(angle_deg: np.ndarray, response: np.ndarray) -> dict:
    if len(angle_deg) == 0 or len(response) == 0:
        return {
            "peak_angle": np.nan,
            "peak_value": np.nan,
            "hpbw": np.nan,
            "sidelobe_level_db": np.nan,
        }

    peak_idx = int(np.argmax(response))
    peak_angle = float(angle_deg[peak_idx])
    peak_value = float(response[peak_idx])

    half_power = peak_value / np.sqrt(2.0)
    above = np.where(response >= half_power)[0]
    if len(above) > 0:
        hpbw = float(angle_deg[above[-1]] - angle_deg[above[0]])
    else:
        hpbw = np.nan

    if np.isnan(hpbw):
        sidelobe_db = np.nan
    else:
        left = peak_angle - hpbw / 2.0
        right = peak_angle + hpbw / 2.0
        mask = (angle_deg < left) | (angle_deg > right)
        outside = response[mask]
        if len(outside) == 0 or peak_value <= 0:
            sidelobe_db = np.nan
        else:
            sll = float(np.max(outside))
            sidelobe_db = 20.0 * np.log10(max(sll, 1e-12) / max(peak_value, 1e-12))

    return {
        "peak_angle": peak_angle,
        "peak_value": peak_value,
        "hpbw": hpbw,
        "sidelobe_level_db": sidelobe_db,
    }


# =============================================================================
# INIT STATE
# =============================================================================
init_state()

linear_animation_angles = list(range(0, 31, 5)) + list(range(25, -31, -5)) + list(range(-25, 0, 5))
circular_animation_phases = np.linspace(0.0, 360.0, 144, endpoint=False)

mode_options = ["Custom grid linear sweep", "Full 9x9 circular orbit", "Circular subarray orbit"]
legacy_mode_map = {
    "circular-lobing-9x9iso": "Full 9x9 circular orbit",
}


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.header("🎛️ Wave Configuration")

freq_unit = st.sidebar.selectbox(
    "Frequency Unit Scale",
    ["0.1 kHz", "1 kHz", "10 kHz"],
    index=["0.1 kHz", "1 kHz", "10 kHz"].index(st.session_state.selected_unit),
)
st.session_state.selected_unit = freq_unit

if freq_unit == "0.1 kHz":
    min_f, max_f, def_f = 0.1, 0.9, 0.5
elif freq_unit == "1 kHz":
    min_f, max_f, def_f = 1.0, 9.0, 1.0
else:
    min_f, max_f, def_f = 10.0, 40.0, 40.0

frequency_khz = st.sidebar.slider("Frequency (kHz)", min_value=min_f, max_value=max_f, value=def_f, step=0.1)
v_sound = 343.0
wavelength_m = v_sound / (frequency_khz * 1000.0)

spacing_factor = st.sidebar.slider(
    "Element Spacing (in units of λ)", min_value=0.1, max_value=3.0, value=0.5, step=0.05
)

current_mode = legacy_mode_map.get(st.session_state.array_mode, st.session_state.array_mode)
array_mode = st.sidebar.selectbox(
    "Array Demo Mode",
    mode_options,
    index=mode_options.index(current_mode) if current_mode in mode_options else 0,
)
if array_mode != st.session_state.array_mode and st.session_state.anim_running:
    st.session_state.anim_running = False
    st.session_state.frame_index = 0
    st.session_state.orbit_trace_x = []
    st.session_state.orbit_trace_y = []
    st.session_state.peak3d_trace_x = []
    st.session_state.peak3d_trace_y = []
    st.session_state.peak3d_trace_z = []
st.session_state.array_mode = array_mode

orbit_radius_deg = st.sidebar.slider(
    "Circular orbit radius (degrees)", min_value=3.0, max_value=25.0, value=12.0, step=1.0
)
orbit_speed = st.sidebar.slider(
    "Orbit speed multiplier", min_value=0.25, max_value=3.0, value=1.0, step=0.25
)
orbit_window_deg = st.sidebar.slider(
    "Orbit map window (degrees)", min_value=20.0, max_value=60.0, value=40.0, step=5.0)
subarray_radius = st.sidebar.slider(
    "Circular subarray radius (grid units)", min_value=2.5, max_value=4.5, value=3.5, step=0.1
)

st.sidebar.header("🎥 3D Camera Angles")
cam_x = st.sidebar.slider("Camera X", min_value=-3.0, max_value=3.0, value=1.25, step=0.1)
cam_y = st.sidebar.slider("Camera Y", min_value=-3.0, max_value=3.0, value=1.25, step=0.1)
cam_z = st.sidebar.slider("Camera Z", min_value=-3.0, max_value=3.0, value=1.25, step=0.1)

custom_camera = {
    "up": dict(x=0, y=0, z=1),
    "center": dict(x=0, y=0, z=0),
    "eye": dict(x=cam_x, y=cam_y, z=cam_z),
}

st.sidebar.header("📈 Visualization Options")
show_phase_map = st.sidebar.checkbox("Show aperture phase map", value=True)
show_metrics = st.sidebar.checkbox("Show beam metrics", value=True)
show_grating_warning = st.sidebar.checkbox("Show spacing warning", value=True)


# =============================================================================
# LAYOUT
# =============================================================================
layout_col_left, layout_col_right = st.columns([1, 4])


# =============================================================================
# LEFT COLUMN: MATRIX PANEL
# =============================================================================
with layout_col_left:
    st.subheader("Matrix")

    grid_coords = np.arange(-4, 5)

    if st.session_state.array_mode == "Full 9x9 circular orbit":
        st.info("Full 9×9 mode locks the aperture to all elements active.")
        st.session_state.grid_state[:, :] = True
    elif st.session_state.array_mode == "Circular subarray orbit":
        st.info("Circular subarray mode locks the aperture to a disk-shaped 9×9 mask.")
        for y_idx, y in enumerate(reversed(grid_coords)):
            for x_idx, x in enumerate(grid_coords):
                st.session_state.grid_state[y_idx, x_idx] = (x * x + y * y) <= (subarray_radius * subarray_radius)

    active_elements = []

    st.markdown('<div class="compact-grid-container">', unsafe_allow_html=True)
    for y_idx, y in enumerate(reversed(grid_coords)):
        st.markdown('<div class="flush-grid-row">', unsafe_allow_html=True)
        cols = st.columns(9)
        for x_idx, x in enumerate(grid_coords):
            with cols[x_idx]:
                is_active = bool(st.session_state.grid_state[y_idx, x_idx])
                button_type = "primary" if is_active else "secondary"
                if st.button(" ", key=f"btn_{x}_{y}", type=button_type):
                    if st.session_state.array_mode == "Custom grid linear sweep":
                        st.session_state.grid_state[y_idx, x_idx] = not is_active
                    st.session_state.anim_running = False
                    st.rerun()

                if st.session_state.grid_state[y_idx, x_idx]:
                    active_elements.append((x, y))
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🧹 Clear Grid", type="secondary", use_container_width=True):
        st.session_state.grid_state = np.zeros((9, 9), dtype=bool)
        st.session_state.anim_running = False
        st.session_state.frame_index = 0
        st.session_state.last_scan_x_deg = 0.0
        st.session_state.last_scan_y_deg = 0.0
        st.session_state.orbit_trace_x = []
        st.session_state.orbit_trace_y = []
        st.session_state.peak3d_trace_x = []
        st.session_state.peak3d_trace_y = []
        st.session_state.peak3d_trace_z = []
        st.rerun()

    if st.session_state.anim_running:
        if st.button("🛑 Stop Animation", type="primary", use_container_width=True):
            st.session_state.anim_running = False
            st.rerun()
    else:
        if st.button("🔄 Animate Beam", type="primary", use_container_width=True):
            st.session_state.anim_running = True
            st.session_state.frame_index = 0
            st.session_state.orbit_trace_x = []
            st.session_state.orbit_trace_y = []
            st.session_state.peak3d_trace_x = []
            st.session_state.peak3d_trace_y = []
            st.session_state.peak3d_trace_z = []
            st.rerun()

    st.markdown("---")
    st.caption("**System Specifications:**")
    st.metric("Wavelength", f"{wavelength_m * 1000:.1f} mm")
    st.metric("Pitch (d)", f"{spacing_factor * wavelength_m * 1000:.1f} mm")
    st.metric("Spacing / λ", f"{spacing_factor:.2f}")


# =============================================================================
# RIGHT COLUMN: SIMULATION
# =============================================================================
with layout_col_right:
    if len(active_elements) > 0:
        if st.session_state.array_mode == "Full 9x9 circular orbit":
            effective_taper = "Gaussian radial"
            effective_spacing_factor = min(spacing_factor, 0.5)
            if spacing_factor > 0.5:
                st.info("Full 9×9 orbit mode clamps spacing to λ/2 for a cleaner beam.")
        elif st.session_state.array_mode == "Circular subarray orbit":
            effective_taper = "Gaussian radial strong"
            effective_spacing_factor = min(spacing_factor, 0.45)
            if spacing_factor > 0.45:
                st.info("Circular subarray mode uses tighter spacing and stronger taper for sidelobe suppression.")
        else:
            effective_taper = taper_mode
            effective_spacing_factor = spacing_factor

        d_spacing = effective_spacing_factor * wavelength_m

        if st.session_state.anim_running:
            if st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit"):
                phase_deg = float(circular_animation_phases[st.session_state.frame_index % len(circular_animation_phases)])
                theta_x_deg = float(orbit_radius_deg * np.cos(np.radians(phase_deg)) * orbit_speed)
                theta_y_deg = float(orbit_radius_deg * np.sin(np.radians(phase_deg)) * orbit_speed)
            else:
                if st.session_state.frame_index >= len(linear_animation_angles):
                    st.session_state.frame_index = 0
                theta_x_deg = float(linear_animation_angles[st.session_state.frame_index])
                theta_y_deg = 0.0
        else:
            theta_x_deg = float(st.session_state.last_scan_x_deg)
            theta_y_deg = float(st.session_state.last_scan_y_deg)

        st.session_state.last_scan_x_deg = theta_x_deg
        st.session_state.last_scan_y_deg = theta_y_deg

        if st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit"):
            scan_x_deg = np.linspace(-orbit_window_deg, orbit_window_deg, 161)
            scan_y_deg = np.linspace(-orbit_window_deg, orbit_window_deg, 161)
            st.info("🏃 Orbit mode is active. The heatmap and traces show the moving main lobe in steering space.")
        else:
            scan_x_deg = np.linspace(-180, 180, 361)
            scan_y_deg = np.linspace(-90, 90, 181)
            if st.session_state.anim_running:
                st.info("🏃 Animation active. Click Stop Animation to freeze on the current frame.")

        azimuth_deg_2d = np.linspace(-180, 180, 361)
        elevation_deg_3d = np.linspace(-90, 90, 181 if not st.session_state.anim_running else 61)
        azimuth_deg_3d = np.linspace(-180, 180, 361 if not st.session_state.anim_running else 121)

        amplitude_2d = compute_array_factor(
            active_elements=active_elements,
            d_spacing=d_spacing,
            wavelength_m=wavelength_m,
            steer_x_deg=theta_x_deg,
            steer_y_deg=theta_y_deg,
            azimuth_deg=azimuth_deg_2d,
            elevation_deg=0.0,
            taper=effective_taper,
        )
        amplitude_matrix = compute_array_factor(
            active_elements=active_elements,
            d_spacing=d_spacing,
            wavelength_m=wavelength_m,
            steer_x_deg=theta_x_deg,
            steer_y_deg=theta_y_deg,
            azimuth_deg=azimuth_deg_3d,
            elevation_deg=elevation_deg_3d,
            taper=effective_taper,
        )

        orbit_map = None
        orbit_peak_x = None
        orbit_peak_y = None
        if st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit"):
            orbit_map = compute_orbit_map(
                active_elements=active_elements,
                d_spacing=d_spacing,
                wavelength_m=wavelength_m,
                steer_x_deg=theta_x_deg,
                steer_y_deg=theta_y_deg,
                scan_x_deg=scan_x_deg,
                scan_y_deg=scan_y_deg,
                taper=effective_taper,
            )
            peak_idx = np.unravel_index(np.argmax(orbit_map), orbit_map.shape)
            orbit_peak_y = float(scan_y_deg[peak_idx[0]])
            orbit_peak_x = float(scan_x_deg[peak_idx[1]])
            if st.session_state.anim_running:
                st.session_state.orbit_trace_x.append(orbit_peak_x)
                st.session_state.orbit_trace_y.append(orbit_peak_y)

        phase_grid = np.full((9, 9), np.nan, dtype=float)
        k = 2.0 * np.pi / wavelength_m
        theta_x_rad = np.radians(theta_x_deg)
        theta_y_rad = np.radians(theta_y_deg)
        for y_idx, y in enumerate(reversed(grid_coords)):
            for x_idx, x in enumerate(grid_coords):
                if st.session_state.grid_state[y_idx, x_idx]:
                    pos_x = x * d_spacing
                    pos_y = y * d_spacing
                    phase_rad = -(k * (pos_x * np.sin(theta_x_rad) + pos_y * np.sin(theta_y_rad)))
                    phase_grid[y_idx, x_idx] = np.degrees(np.angle(np.exp(1j * phase_rad)))

        peak3d_idx = np.unravel_index(np.argmax(amplitude_matrix), amplitude_matrix.shape)
        peak3d_el = float(elevation_deg_3d[peak3d_idx[0]])
        peak3d_az = float(azimuth_deg_3d[peak3d_idx[1]])
        peak3d_gain = float(amplitude_matrix[peak3d_idx])
        if st.session_state.anim_running:
            st.session_state.peak3d_trace_x.append(peak3d_az)
            st.session_state.peak3d_trace_y.append(peak3d_el)
            st.session_state.peak3d_trace_z.append(peak3d_gain)

        metrics = estimate_metrics(azimuth_deg_2d, amplitude_2d)
        grating_flag = st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit") and spacing_factor > 0.5

        if st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit"):
            fig_main = go.Figure(
                data=[
                    go.Heatmap(
                        x=scan_x_deg,
                        y=scan_y_deg,
                        z=orbit_map,
                        colorscale="Viridis",
                        colorbar=dict(title="Gain"),
                        hoverongaps=False,
                    )
                ]
            )
            fig_main.add_trace(
                go.Scatter(
                    x=st.session_state.orbit_trace_x,
                    y=st.session_state.orbit_trace_y,
                    mode="lines+markers",
                    marker=dict(size=7),
                    line=dict(width=2),
                    name="Actual peak trace",
                )
            )
            fig_main.add_trace(
                go.Scatter(
                    x=orbit_radius_deg * orbit_speed * np.cos(np.radians(circular_animation_phases)),
                    y=orbit_radius_deg * orbit_speed * np.sin(np.radians(circular_animation_phases)),
                    mode="lines",
                    line=dict(width=2, dash="dash"),
                    name="Target orbit",
                )
            )
            fig_main.add_trace(
                go.Scatter(
                    x=[orbit_peak_x] if orbit_peak_x is not None else [],
                    y=[orbit_peak_y] if orbit_peak_y is not None else [],
                    mode="markers",
                    marker=dict(size=12, symbol="circle-open"),
                    name="Current peak",
                )
            )
            fig_main.update_layout(
                title=dict(text="Circular-Lobe Scan Map", font=dict(size=16)),
                xaxis_title="X Steering (°)",
                yaxis_title="Y Steering (°)",
                template="plotly_dark",
                height=520,
                margin=dict(l=40, r=40, t=50, b=40),
                xaxis=dict(scaleanchor="y", scaleratio=1),
            )
        else:
            fig_main = go.Figure(
                go.Scatter(
                    x=azimuth_deg_2d,
                    y=amplitude_2d,
                    mode="lines",
                    line=dict(color="#00CC96", width=3, shape="spline"),
                    name="Main Beam Cut",
                )
            )
            fig_main.update_layout(
                title=dict(text="Directivity Pattern Cut-Plane (Elevation = 0°)", font=dict(size=16)),
                xaxis_title="Azimuth Angle (Degrees)",
                yaxis_title="Normalized Array Gain",
                template="plotly_dark",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
                yaxis=dict(range=[0, 1.05]),
            )

        fig_3d = go.Figure(
            data=[
                go.Surface(
                    z=amplitude_matrix,
                    x=azimuth_deg_3d,
                    y=elevation_deg_3d,
                    colorscale="Viridis",
                    lighting=dict(ambient=0.6, roughness=0.4, diffuse=0.8),
                    colorbar=dict(title="Gain", thickness=15),
                )
            ]
        )
        fig_3d.add_trace(
            go.Scatter3d(
                x=st.session_state.peak3d_trace_x,
                y=st.session_state.peak3d_trace_y,
                z=st.session_state.peak3d_trace_z,
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(width=4),
                name="Peak trace",
            )
        )
        fig_3d.add_trace(
            go.Scatter3d(
                x=[peak3d_az],
                y=[peak3d_el],
                z=[peak3d_gain],
                mode="markers",
                marker=dict(size=7),
                name="Current peak",
            )
        )
        fig_3d.update_layout(
            title=dict(text="Complete 3D Spatial Radiation Topography", font=dict(size=16)),
            scene=dict(
                xaxis_title="Azimuth (°)",
                yaxis_title="Elevation (°)",
                zaxis_title="Array Response",
                aspectmode="manual",
                aspectratio=dict(x=1.5, y=1.5, z=1.0),
                camera=custom_camera,
            ),
            template="plotly_dark",
            height=850,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        if show_phase_map:
            phase_fig = go.Figure(
                data=go.Heatmap(
                    z=phase_grid,
                    x=grid_coords,
                    y=list(reversed(grid_coords)),
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="Phase (°)"),
                    hoverongaps=False,
                )
            )
            phase_fig.update_layout(
                title=dict(text="Aperture Phase Map", font=dict(size=16)),
                xaxis_title="X Element Index",
                yaxis_title="Y Element Index",
                template="plotly_dark",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
            )

        st.markdown(
            f"### ⚡ Current Electronic Steering Vector: `x={theta_x_deg:.1f}°`, `y={theta_y_deg:.1f}°`"
            if st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit")
            else f"### ⚡ Current Electronic Steering Angle: `{theta_x_deg:.1f}°`"
        )

        if show_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peak Angle", f"{metrics['peak_angle']:.1f}°")
            m2.metric("Peak Gain", f"{metrics['peak_value']:.2f}")
            m3.metric("HPBW", f"{metrics['hpbw']:.1f}°" if not np.isnan(metrics["hpbw"]) else "n/a")
            if np.isnan(metrics["sidelobe_level_db"]):
                m4.metric("Sidelobe", "n/a")
            else:
                m4.metric("Sidelobe", f"{metrics['sidelobe_level_db']:.1f} dB")

        if show_grating_warning:
            if grating_flag:
                st.warning("Spacing is above λ/2, so grating lobes can appear. The orbit modes clamp spacing to help keep the beam cleaner.")
            elif st.session_state.array_mode == "Full 9x9 circular orbit":
                st.success("Full 9×9 orbit mode is using a Gaussian radial taper and a visible peak trace.")
            elif st.session_state.array_mode == "Circular subarray orbit":
                st.success("Circular subarray mode uses a disk-shaped aperture plus a stronger Gaussian taper to suppress sidelobes.")
            elif spacing_factor > 0.5:
                st.warning("Element spacing is above λ/2, so grating lobes may appear as the beam steers away from broadside.")

        st.plotly_chart(fig_main, use_container_width=True, key="main_beam_plot")

        if show_phase_map:
            st.plotly_chart(phase_fig, use_container_width=True, key="phase_map")

        st.plotly_chart(fig_3d, use_container_width=True, key="static_layout_p3d")

        if st.session_state.anim_running:
            time.sleep(0.08)
            if st.session_state.array_mode in ("Full 9x9 circular orbit", "Circular subarray orbit"):
                st.session_state.frame_index = (st.session_state.frame_index + 1) % len(circular_animation_phases)
            else:
                st.session_state.frame_index = (st.session_state.frame_index + 1) % len(linear_animation_angles)
            st.rerun()
    else:
        st.warning("Awaiting selection matrix input. Toggle array elements on the left grid panel to begin simulation execution.")

