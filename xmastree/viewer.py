"""
Christmas Tree Atomic Viewer
Custom 3D viewer for ASE Atoms with inline dependencies.

Colors:
- Ag: Silver (#C0C0C0)
- Fe: Brown (#8B4513)
- Au: Gold (#FFD700)
- C: Green (#228B22)
- H: White (#FFFFFF)
- F: Light green (#90E050)
- Cl: Green (#1FF01F)
- Br: Dark red (#A62929)
"""

import json
import os
from pathlib import Path
from IPython.display import HTML
from typing import List, Union, Optional
from ase import Atoms
from ase.io.trajectory import TrajectoryReader
from ase.calculators.calculator import PropertyNotImplementedError


# Christmas tree specific colors and radii
ATOM_COLORS = {
    'H': '#FFFFFF',   # White
    'C': '#228B22',   # Forest Green (leaves)
    'I': '#940094',   # Purple (decoration - iodine)
    'Cl': '#1FF01F',  # Green (decoration)
    'Br': '#A62929',  # Dark red (decoration)
    'Fe': '#8B4513',  # Saddle Brown (trunk)
    'Ag': '#C0C0C0',  # Silver (base)
    'Au': '#FFD700',  # Gold (star)
    'default': '#FFC0CB'
}

ATOM_RADII = {
    'H': 0.31,
    'C': 0.76,
    'I': 1.39,        # Iodine (larger than Br)
    'Cl': 1.02,
    'Br': 1.20,
    'Fe': 1.52,
    'Ag': 1.80,       # Larger silver atoms for base
    'Au': 1.36,
    'default': 0.7
}


def _load_library(filename: str) -> str:
    """Load a JavaScript library from the libs folder."""
    lib_path = Path(__file__).parent / "libs" / filename
    if lib_path.exists():
        return lib_path.read_text(encoding='utf-8')
    else:
        raise FileNotFoundError(f"Library not found: {lib_path}")


def viewer(
    atoms_or_traj: Union[Atoms, List[Atoms], TrajectoryReader],
    rotation_mode: str = 'orbit',
    bg_color: str = '#1a1a2e',
    show_bonds: bool = True,
    show_forces: bool = False,
    show_cell: bool = False,
    show_energy: bool = True,
    atom_scale: float = 0.8,
    bond_cutoff_scale: float = 1.2,
    bond_thickness: float = 0.08,
    force_scale: float = 0.3,
    animation_fps: int = 10,
    width: int = 800,
    height: int = 600,
    write_html: Optional[str] = None,
) -> HTML:
    """
    Visualize ASE Atoms or trajectory in 3D with inline dependencies.

    Parameters:
        atoms_or_traj: Single Atoms object, list of Atoms, or TrajectoryReader
        rotation_mode: 'orbit' or 'trackball'
        bg_color: Background color (hex)
        show_bonds: Show bonds between atoms
        show_forces: Show force vectors
        show_cell: Show unit cell
        show_energy: Show energy plot (if available)
        atom_scale: Scale factor for atom sizes
        bond_cutoff_scale: Cutoff scale for bond detection
        bond_thickness: Thickness of bonds
        force_scale: Scale factor for force vectors
        animation_fps: Animation frames per second
        width: Viewer width in pixels
        height: Viewer height in pixels
        write_html: Optional filename to save HTML

    Returns:
        IPython HTML object for display
    """
    # Convert input to list of frames
    if isinstance(atoms_or_traj, Atoms):
        frames = [atoms_or_traj]
    else:
        frames = [atoms for atoms in atoms_or_traj]

    # Extract trajectory data
    trajectory_data = []
    has_energy = False
    for atoms in frames:
        frame_data = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions().tolist(),
            "cell": atoms.get_cell().tolist(),
        }
        try:
            forces = atoms.get_forces(apply_constraint=False)
            frame_data["forces"] = forces.tolist()
        except (RuntimeError, PropertyNotImplementedError):
            pass
        try:
            energy = atoms.get_potential_energy()
            frame_data["energy"] = energy
            has_energy = True
        except (RuntimeError, PropertyNotImplementedError):
            frame_data["energy"] = None
        trajectory_data.append(frame_data)

    json_data = json.dumps(trajectory_data)
    effective_show_energy = show_energy and has_energy

    # Build atom info for JavaScript
    atom_info_parts = []
    for symbol in ATOM_RADII:
        if symbol == 'default':
            continue
        radius = ATOM_RADII[symbol]
        hex_color = ATOM_COLORS.get(symbol, ATOM_COLORS['default'])
        color_int_str = f"0x{hex_color.lstrip('#')}"
        atom_info_parts.append(f"'{symbol}':{{color:{color_int_str},radius:{radius}}}")

    # Add default
    default_radius = ATOM_RADII['default']
    default_hex_color = ATOM_COLORS['default']
    default_color_int_str = f"0x{default_hex_color.lstrip('#')}"
    atom_info_parts.append(f"'default':{{color:{default_color_int_str},radius:{default_radius}}}")

    atom_info_js = ", ".join(atom_info_parts)

    # Load JavaScript libraries
    three_js = _load_library("three.min.js")
    orbit_controls_js = _load_library("OrbitControls.js")
    trackball_controls_js = _load_library("TrackballControls.js")
    chart_js = _load_library("chart.min.js")
    gif_js = _load_library("gif.js")
    gif_worker_js = _load_library("gif.worker.js")

    # Escape gif worker for embedding as string
    gif_worker_escaped = gif_worker_js.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Christmas Tree Viewer</title>
        <style>
            body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; overflow: hidden; user-select: none; }}
            .main-container {{ display: flex; flex-direction: column; width: {width}px; background-color: #1A202C; border-radius: 8px; border: 1px solid #4A5568; }}
            .viewer-wrapper {{ position: relative; width: 100%; height: {height}px; overflow: hidden; }}
            .viewer-container {{ width: 100%; height: 100%; display: flex; }}
            .sidebar {{
                width: 240px; min-width: 240px; padding: 20px;
                background-color: #1A202C; color: #E2E8F0;
                display: flex; flex-direction: column; gap: 20px;
                overflow-y: auto; overflow-x: hidden;
                transition: width 0.3s ease, min-width 0.3s ease, padding 0.3s ease;
                flex-shrink: 0; border-right: 1px solid #4A5568;
            }}
            .viewer-wrapper.sidebar-collapsed .sidebar {{ width: 0; min-width: 0; padding: 20px 0; border-right-color: transparent; }}
            #sidebar-toggle-btn {{
                position: absolute; top: 10px; left: 10px; z-index: 1000;
                background-color: #2D3748; color: white; border: 1px solid #4A5568;
                border-radius: 6px; width: 30px; height: 30px;
                display: flex; align-items: center; justify-content: center;
                cursor: pointer; font-size: 16px; font-weight: bold;
                font-family: "Courier New", Courier, monospace;
                transition: background-color 0.2s;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            #sidebar-toggle-btn:hover {{ background-color: #4A5568; }}
            #top-right-controls {{ position: absolute; top: 15px; right: 15px; z-index: 100; }}
            #save-gif-btn {{
                background-color: rgba(74, 85, 104, 0.7); border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px; color: #E2E8F0; padding: 10px 16px; cursor: pointer;
                font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 8px;
                transition: all 0.3s ease; backdrop-filter: blur(5px);
            }}
            #save-gif-btn:hover {{ background-color: rgba(99, 110, 130, 0.85); transform: translateY(-2px); }}
            #save-gif-btn.saving {{ background: #6B7280; cursor: not-allowed; transform: none; }}
            @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            .spinner {{ animation: spin 1s linear infinite; }}
            .sidebar h2 {{ font-size: 18px; margin: 0 0 10px 0; padding-bottom: 10px; border-bottom: 1px solid #4A5568; color: #A0AEC0; white-space: nowrap; }}
            details {{ color: #E2E8F0; border-bottom: 1px solid #2d3748; white-space: nowrap; }}
            details[open] {{ padding-bottom: 15px; }}
            summary {{ cursor: pointer; font-size: 1rem; font-weight: bold; padding: 10px 0; list-style-position: inside; }}
            summary::marker {{ color: #A0AEC0; }}
            .details-content {{ padding-top: 10px; display: flex; flex-direction: column; gap: 12px; }}
            .toggle-switch {{ display: flex; justify-content: space-between; align-items: center; width: 100%; font-size: 14px; }}
            .toggle-switch input {{ display: none; }}
            .toggle-switch .slider {{ position: relative; cursor: pointer; width: 40px; height: 22px; background-color: #ccc; border-radius: 22px; transition: background-color 0.2s; flex-shrink: 0; }}
            .toggle-switch .slider:before {{ position: absolute; content: ""; height: 16px; width: 16px; left: 3px; bottom: 3px; background-color: white; border-radius: 50%; transition: transform 0.2s; }}
            .toggle-switch input:checked + .slider {{ background-color: #4CAF50; }}
            .toggle-switch input:checked + .slider:before {{ transform: translateX(18px); }}
            select {{ background-color: #2D3748; color: #E2E8F0; border: 1px solid #4A5568; border-radius: 4px; padding: 4px; width: 100%; }}
            input[type="color"] {{ width: 100%; height: 25px; border: 1px solid #4A5568; border-radius: 4px; padding: 2px; background-color: #2D3748; }}
            .animation-controls {{ display: flex; align-items: center; gap: 10px; }}
            .control-btn {{ background-color: #4A5568; border: none; border-radius: 50%; width: 32px; height: 32px; cursor: pointer; display: flex; justify-content: center; align-items: center; flex-shrink: 0; transition: background-color 0.2s; color: white; }}
            .control-btn:hover {{ background-color: #636e82; }}
            .control-btn svg {{ fill: white; }}
            input[type="range"] {{ width: 100%; }}
            .slider-group {{ display: flex; flex-direction: column; gap: 4px; }}
            .slider-label, .slider-value {{ font-size: 12px; color: #A0AEC0; }}
            #canvas-container {{ flex-grow: 1; height: 100%; cursor: grab; background-color: {bg_color}; min-width: 0; position: relative; }}
            #plot-container {{ width: 100%; padding: 10px; box-sizing: border-box; background-color: #1A202C; display: {'block' if effective_show_energy else 'none'}; }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="viewer-wrapper" id="viewer-wrapper">
                <div class="viewer-container">
                    <div class="sidebar" id="sidebar">
                        <h2 style="text-align: right;">Christmas Tree</h2>
                        <details open><summary>Display Options</summary><div class="details-content">
                            <label class="toggle-switch"><span>Bonds</span><input type="checkbox" id="bonds-toggle"><span class="slider"></span></label>
                            <label class="toggle-switch"><span>Forces</span><input type="checkbox" id="forces-toggle"><span class="slider"></span></label>
                            <label class="toggle-switch"><span>Unit Cell</span><input type="checkbox" id="cell-toggle"><span class="slider"></span></label>
                            <label class="toggle-switch"><span>Shadows</span><input type="checkbox" id="shadows-toggle"><span class="slider"></span></label>
                            <label class="toggle-switch" id="energy-toggle-label" style="display: {'flex' if has_energy else 'none'};"><span>Energy Plot</span><input type="checkbox" id="energy-plot-toggle"><span class="slider"></span></label>
                        </div></details>
                        <details open><summary>Advanced Settings</summary><div class="details-content">
                            <div class="slider-group"><label class="slider-label">Rotation Mode</label><select id="rotation-mode-select"><option value="orbit">Orbit</option><option value="trackball">Trackball</option></select></div>
                            <div class="slider-group"><label class="slider-label">Style</label><select id="style-select"><option value="matte">Matte</option><option value="glossy">Glossy</option><option value="metallic">Metallic</option><option value="toon">Toon</option><option value="neon">Neon</option></select></div>
                            <div class="slider-group"><label class="slider-label">Atom Size</label><input type="range" id="atom-scale-slider" min="0.1" max="2.0" step="0.05" value="{atom_scale}"><span id="atom-scale-label" class="slider-value">Scale: {atom_scale:.2f}</span></div>
                            <div class="slider-group"><label class="slider-label">Bond Thickness</label><input type="range" id="bond-thickness-slider" min="0.02" max="0.2" step="0.01" value="{bond_thickness}"><span id="bond-thickness-label" class="slider-value">Value: {bond_thickness:.2f}</span></div>
                            <div class="slider-group"><label class="slider-label">Bond Cutoff Scale</label><input type="range" id="bond-cutoff-slider" min="0.8" max="1.5" step="0.01" value="{bond_cutoff_scale}"><span id="bond-cutoff-label" class="slider-value">Scale: {bond_cutoff_scale:.2f}</span></div>
                            <div class="slider-group"><label class="slider-label">Animation Speed (FPS)</label><input type="range" id="animation-speed-slider" min="1" max="60" step="1" value="{animation_fps}"><span id="animation-speed-label" class="slider-value">FPS: {animation_fps}</span></div>
                            <div class="slider-group"><label class="slider-label">Background Color</label><input type="color" id="bg-color-picker" value="{bg_color}"></div>
                        </div></details>
                        <div id="animation-section" style="padding-top: 10px; display: none;"><h3 style="margin:0 0 10px 0; color: #A0AEC0;">Animation</h3><div class="animation-controls">
                            <button id="play-pause-btn" class="control-btn" title="Play/Pause"><svg id="play-icon" width="14" height="14" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><svg id="pause-icon" width="14" height="14" viewBox="0 0 24 24" style="display:none;"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg></button>
                            <input type="range" id="frame-slider" min="0" value="0">
                            <button id="copy-xyz-btn" class="control-btn" title="Copy XYZ"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg></button>
                        </div><span id="frame-label" class="slider-value">Frame: 0 / 0</span></div>
                    </div>
                    <div id="canvas-container"></div>
                </div>
                <div id="sidebar-toggle-btn">&#10094;</div>
                <div id="top-right-controls">
                    <button id="save-gif-btn" title="Save as GIF">
                        <span id="gif-icon">&#128190;</span>
                        <span id="gif-text">Save GIF</span>
                    </button>
                </div>
            </div>
            <div id="plot-container"><canvas id="energy-plot"></canvas></div>
        </div>

        <script>{three_js}</script>
        <script>{orbit_controls_js}</script>
        <script>{trackball_controls_js}</script>
        <script>{chart_js}</script>
        <script>{gif_js}</script>

        <script>
            const gifWorkerCode = `{gif_worker_escaped}`;
            const trajectoryData = {json_data};
            const hasEnergy = {'true' if has_energy else 'false'};
            const initialShowBonds = {str(show_bonds).lower()};
            const initialShowForces = {str(show_forces).lower()};
            const initialShowCell = {str(show_cell).lower()};
            const initialShowEnergy = {str(effective_show_energy).lower()};
            const atomInfo = {{{atom_info_js}}};

            let scene, camera, renderer, controls, energyPlot, ambientLight, directionalLight;
            let atomGroup, bondGroup, forceGroup, cellGroup;
            let animationInterval = null, isPlaying = false;
            let currentFrameIndex = 0;
            let shadowsEnabled = false;
            let currentStyle = 'matte';
            let currentAtomScale = {atom_scale};
            let currentBondCutoffFactor = {bond_cutoff_scale};
            let currentBondThickness = {bond_thickness};

            init();

            function init() {{
                const container = document.getElementById('canvas-container');
                scene = new THREE.Scene();
                scene.background = new THREE.Color(document.getElementById('bg-color-picker').value);
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.z = 30;
                scene.add(camera);

                renderer = new THREE.WebGLRenderer({{ antialias: true, preserveDrawingBuffer: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 10, 7);
                scene.add(directionalLight);

                atomGroup = new THREE.Group();
                bondGroup = new THREE.Group();
                forceGroup = new THREE.Group();
                cellGroup = new THREE.Group();
                scene.add(atomGroup, bondGroup, forceGroup, cellGroup);

                setupControls(document.getElementById('rotation-mode-select').value);
                setupUI();
                if (hasEnergy) setupEnergyPlot();
                window.addEventListener('resize', onWindowResize);
                updateScene(0);
                animate();
            }}

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}

            function onWindowResize() {{
                const container = document.getElementById('canvas-container');
                if (!container || container.clientWidth === 0) return;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }}

            function setupControls(mode) {{
                if (controls) controls.dispose();
                const container = document.getElementById('canvas-container');
                if (mode === 'trackball') {{
                    controls = new THREE.TrackballControls(camera, container);
                }} else {{
                    controls = new THREE.OrbitControls(camera, container);
                    controls.enableDamping = true;
                }}
            }}

            function setupUI() {{
                const viewerWrapper = document.getElementById('viewer-wrapper');
                const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');

                sidebarToggleBtn.addEventListener('click', () => {{
                    const isCollapsed = viewerWrapper.classList.toggle('sidebar-collapsed');
                    sidebarToggleBtn.innerHTML = isCollapsed ? '&#10095;' : '&#10094;';
                    setTimeout(onWindowResize, 310);
                }});

                document.getElementById('rotation-mode-select').value = "{rotation_mode}";
                document.getElementById('rotation-mode-select').addEventListener('change', (e) => setupControls(e.target.value));
                document.getElementById('bg-color-picker').addEventListener('input', (e) => scene.background.set(e.target.value));

                if (trajectoryData.length > 1) {{
                    document.getElementById('animation-section').style.display = 'block';
                    const slider = document.getElementById('frame-slider');
                    slider.max = trajectoryData.length - 1;
                    slider.addEventListener('input', (e) => updateScene(parseInt(e.target.value)));
                    document.getElementById('play-pause-btn').addEventListener('click', togglePlay);
                    document.getElementById('copy-xyz-btn').addEventListener('click', copyXyzToClipboard);
                }}

                document.getElementById('save-gif-btn').addEventListener('click', saveAsGif);

                const bondsToggle = document.getElementById('bonds-toggle');
                const forcesToggle = document.getElementById('forces-toggle');
                const cellToggle = document.getElementById('cell-toggle');
                const shadowsToggle = document.getElementById('shadows-toggle');

                bondsToggle.checked = initialShowBonds; bondGroup.visible = initialShowBonds;
                forcesToggle.checked = initialShowForces; forceGroup.visible = initialShowForces;
                cellToggle.checked = initialShowCell; cellGroup.visible = initialShowCell;

                bondsToggle.addEventListener('change', (e) => {{ bondGroup.visible = e.target.checked; }});
                forcesToggle.addEventListener('change', (e) => {{ forceGroup.visible = e.target.checked; }});
                cellToggle.addEventListener('change', (e) => {{ cellGroup.visible = e.target.checked; }});
                shadowsToggle.addEventListener('change', (e) => {{ shadowsEnabled = e.target.checked; updateScene(currentFrameIndex, false); }});

                if (hasEnergy) {{
                    const energyToggle = document.getElementById('energy-plot-toggle');
                    energyToggle.checked = initialShowEnergy;
                    document.getElementById('plot-container').style.display = initialShowEnergy ? 'block' : 'none';
                    energyToggle.addEventListener('change', (e) => {{
                        document.getElementById('plot-container').style.display = e.target.checked ? 'block' : 'none';
                    }});
                }}

                document.getElementById('atom-scale-slider').addEventListener('input', (e) => {{
                    currentAtomScale = parseFloat(e.target.value);
                    document.getElementById('atom-scale-label').textContent = `Scale: ${{currentAtomScale.toFixed(2)}}`;
                    updateScene(currentFrameIndex, false);
                }});
                document.getElementById('bond-cutoff-slider').addEventListener('input', (e) => {{
                    currentBondCutoffFactor = parseFloat(e.target.value);
                    document.getElementById('bond-cutoff-label').textContent = `Scale: ${{currentBondCutoffFactor.toFixed(2)}}`;
                    updateScene(currentFrameIndex, false);
                }});
                document.getElementById('bond-thickness-slider').addEventListener('input', (e) => {{
                    currentBondThickness = parseFloat(e.target.value);
                    document.getElementById('bond-thickness-label').textContent = `Value: ${{currentBondThickness.toFixed(2)}}`;
                    updateScene(currentFrameIndex, false);
                }});
                document.getElementById('animation-speed-slider').addEventListener('input', (e) => {{
                    document.getElementById('animation-speed-label').textContent = `FPS: ${{e.target.value}}`;
                    if (isPlaying) {{
                        clearInterval(animationInterval);
                        const speed = 1000 / parseInt(e.target.value);
                        animationInterval = setInterval(() => {{
                            let nextFrame = (currentFrameIndex + 1) % trajectoryData.length;
                            updateScene(nextFrame);
                        }}, speed);
                    }}
                }});
                document.getElementById('style-select').addEventListener('change', (e) => {{
                    currentStyle = e.target.value;
                    updateScene(currentFrameIndex, false);
                }});
            }}

            function togglePlay() {{
                isPlaying = !isPlaying;
                const playIcon = document.getElementById('play-icon');
                const pauseIcon = document.getElementById('pause-icon');
                if (isPlaying) {{
                    playIcon.style.display = 'none';
                    pauseIcon.style.display = 'block';
                    const speed = 1000 / parseInt(document.getElementById('animation-speed-slider').value);
                    animationInterval = setInterval(() => {{
                        let nextFrame = (currentFrameIndex + 1) % trajectoryData.length;
                        updateScene(nextFrame);
                    }}, speed);
                }} else {{
                    playIcon.style.display = 'block';
                    pauseIcon.style.display = 'none';
                    clearInterval(animationInterval);
                }}
            }}

            function updateScene(frameIdx, updateSlider = true) {{
                currentFrameIndex = frameIdx;
                clearGroup(atomGroup); clearGroup(bondGroup); clearGroup(forceGroup); clearGroup(cellGroup);

                const frameData = trajectoryData[currentFrameIndex];
                if (!frameData) return;

                const positions = frameData.positions.map(p => new THREE.Vector3(...p));
                const symbols = frameData.symbols;

                positions.forEach((pos, i) => {{
                    const atom = createAtom(pos, symbols[i]);
                    if (atom) atomGroup.add(atom);
                }});

                if (document.getElementById('bonds-toggle').checked) {{
                    for (let i = 0; i < positions.length; i++) {{
                        for (let j = i + 1; j < positions.length; j++) {{
                            const r_i = (atomInfo[symbols[i]] || atomInfo['default']).radius;
                            const r_j = (atomInfo[symbols[j]] || atomInfo['default']).radius;
                            const cutoff = (r_i + r_j) * currentBondCutoffFactor;
                            if (positions[i].distanceTo(positions[j]) < cutoff) {{
                                createBond(positions[i], positions[j], symbols[i], symbols[j]);
                            }}
                        }}
                    }}
                }}

                if (document.getElementById('cell-toggle').checked && frameData.cell && frameData.cell.flat().some(v => v !== 0)) {{
                    drawCell(frameData.cell);
                }}

                if (updateSlider && trajectoryData.length > 1) {{
                    document.getElementById('frame-slider').value = currentFrameIndex;
                    document.getElementById('frame-label').textContent = `Frame: ${{currentFrameIndex}} / ${{trajectoryData.length - 1}}`;
                }}
                if (hasEnergy) updatePlotHighlight(currentFrameIndex);
            }}

            function createAtom(pos, symbol) {{
                const info = atomInfo[symbol] || atomInfo['default'];
                const scaledRadius = info.radius * currentAtomScale;
                let geometry, material, sphere;

                switch (currentStyle) {{
                    case 'toon':
                        geometry = new THREE.SphereGeometry(scaledRadius, 32, 32);
                        material = new THREE.MeshToonMaterial({{ color: info.color }});
                        sphere = new THREE.Mesh(geometry, material);
                        sphere.position.copy(pos);
                        const outlineGeo = new THREE.SphereGeometry(scaledRadius * 1.08, 32, 32);
                        const outlineMat = new THREE.MeshBasicMaterial({{ color: 0x000000, side: THREE.BackSide }});
                        const outline = new THREE.Mesh(outlineGeo, outlineMat);
                        outline.position.copy(pos);
                        const group = new THREE.Group();
                        group.add(outline); group.add(sphere);
                        return group;
                    case 'neon':
                        geometry = new THREE.SphereGeometry(scaledRadius, 32, 32);
                        material = new THREE.MeshBasicMaterial({{ color: new THREE.Color(info.color).multiplyScalar(1.5) }});
                        sphere = new THREE.Mesh(geometry, material);
                        sphere.position.copy(pos);
                        return sphere;
                    case 'glossy':
                        geometry = new THREE.SphereGeometry(scaledRadius, 32, 32);
                        material = new THREE.MeshPhongMaterial({{ color: info.color, shininess: 100 }});
                        sphere = new THREE.Mesh(geometry, material);
                        sphere.position.copy(pos);
                        return sphere;
                    case 'metallic':
                        geometry = new THREE.SphereGeometry(scaledRadius, 32, 32);
                        material = new THREE.MeshStandardMaterial({{ color: info.color, metalness: 0.5, roughness: 0.3 }});
                        sphere = new THREE.Mesh(geometry, material);
                        sphere.position.copy(pos);
                        return sphere;
                    default:
                        geometry = new THREE.SphereGeometry(scaledRadius, 32, 32);
                        material = new THREE.MeshLambertMaterial({{ color: info.color }});
                        sphere = new THREE.Mesh(geometry, material);
                        sphere.position.copy(pos);
                        return sphere;
                }}
            }}

            function createBond(p1, p2, sym1, sym2) {{
                const midPoint = p1.clone().add(p2).multiplyScalar(0.5);
                const bond1 = createHalfBond(p1, midPoint, (atomInfo[sym1] || atomInfo.default).color);
                const bond2 = createHalfBond(midPoint, p2, (atomInfo[sym2] || atomInfo.default).color);
                if (bond1) bondGroup.add(bond1);
                if (bond2) bondGroup.add(bond2);
            }}

            function createHalfBond(start, end, color) {{
                if (start.distanceTo(end) <= 0) return null;
                const path = new THREE.LineCurve3(start, end);
                const geometry = new THREE.TubeGeometry(path, 1, currentBondThickness, 8, false);
                let material;
                if (currentStyle === 'neon') {{
                    material = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
                }} else if (currentStyle === 'glossy') {{
                    material = new THREE.MeshPhongMaterial({{ color: color, shininess: 100 }});
                }} else if (currentStyle === 'metallic') {{
                    material = new THREE.MeshStandardMaterial({{ color: color, metalness: 0.3, roughness: 0.4 }});
                }} else {{
                    material = new THREE.MeshLambertMaterial({{ color: color }});
                }}
                return new THREE.Mesh(geometry, material);
            }}

            function clearGroup(group) {{
                while (group.children.length > 0) {{
                    const child = group.children[0];
                    group.remove(child);
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {{
                        if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                        else if (child.material.dispose) child.material.dispose();
                    }}
                    if (child.children && child.children.length > 0) clearGroup(child);
                }}
            }}

            function drawCell(cell) {{
                const [v1, v2, v3] = cell.map(c => new THREE.Vector3(...c));
                const points = [
                    new THREE.Vector3(0,0,0), v1, v2, v3,
                    v1.clone().add(v2), v1.clone().add(v3), v2.clone().add(v3),
                    v1.clone().add(v2).add(v3)
                ];
                const edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
                const lineMat = new THREE.LineBasicMaterial({{ color: 0x888888 }});
                edges.forEach(edge => {{
                    const geo = new THREE.BufferGeometry().setFromPoints([points[edge[0]], points[edge[1]]]);
                    cellGroup.add(new THREE.Line(geo, lineMat));
                }});
            }}

            function setupEnergyPlot() {{
                const ctx = document.getElementById('energy-plot').getContext('2d');
                const energyData = trajectoryData.map((frame, i) => ({{ x: i, y: frame.energy }}));
                energyPlot = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [
                            {{ label: 'Energy', data: energyData, backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', showLine: true, pointRadius: 3 }},
                            {{ label: 'Current', data: [], backgroundColor: 'rgba(255, 99, 132, 1)', pointRadius: 6, pointStyle: 'rectRot' }}
                        ]
                    }},
                    options: {{
                        scales: {{
                            x: {{ title: {{ display: true, text: 'Frame', color: '#A0AEC0' }}, ticks: {{ color: '#A0AEC0' }}, grid: {{ color: '#4A5568' }} }},
                            y: {{ title: {{ display: true, text: 'Energy [eV]', color: '#A0AEC0' }}, ticks: {{ color: '#A0AEC0' }}, grid: {{ color: '#4A5568' }} }}
                        }},
                        plugins: {{ legend: {{ display: false }} }},
                        onClick: (e, elements) => {{ if (elements.length > 0 && elements[0].datasetIndex === 0) updateScene(elements[0].index); }}
                    }}
                }});
            }}

            function updatePlotHighlight(frameIndex) {{
                if (!energyPlot || !trajectoryData[frameIndex] || trajectoryData[frameIndex].energy === null) return;
                energyPlot.data.datasets[1].data = [{{ x: frameIndex, y: trajectoryData[frameIndex].energy }}];
                energyPlot.update('none');
            }}

            function copyXyzToClipboard() {{
                const frameData = trajectoryData[currentFrameIndex];
                if (!frameData) return;
                const nAtoms = frameData.symbols.length;
                const comment = `Frame ${{currentFrameIndex}}, Energy = ${{frameData.energy ? frameData.energy.toFixed(6) : "N/A"}} eV`;
                let xyzString = `${{nAtoms}}\\n${{comment}}\\n`;
                for (let i = 0; i < nAtoms; i++) {{
                    const [x, y, z] = frameData.positions[i];
                    xyzString += `${{frameData.symbols[i].padEnd(4)}} ${{x.toFixed(8).padStart(12)}} ${{y.toFixed(8).padStart(12)}} ${{z.toFixed(8).padStart(12)}}\\n`;
                }}
                navigator.clipboard.writeText(xyzString).then(() => {{
                    const btn = document.getElementById('copy-xyz-btn');
                    btn.innerHTML = '&#10003;';
                    setTimeout(() => {{ btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'; }}, 2000);
                }});
            }}
            <!-- 
            ========================================
            async function saveAsGif() {{
                const btn = document.getElementById('save-gif-btn');
                const icon = document.getElementById('gif-icon');
                const text = document.getElementById('gif-text');
                btn.classList.add('saving');
                icon.innerHTML = '&#8987;';
                icon.classList.add('spinner');
                text.textContent = 'Recording...';
                btn.disabled = true;

                try {{
                    const workerBlob = new Blob([gifWorkerCode], {{ type: 'application/javascript' }});
                    const workerUrl = URL.createObjectURL(workerBlob);

                    const wasPlaying = isPlaying;
                    if (wasPlaying) togglePlay();

                    const gif = new GIF({{
                        workers: 2, quality: 10,
                        width: renderer.domElement.width, height: renderer.domElement.height,
                        workerScript: workerUrl
                    }});

                    gif.on('finished', function(blob) {{
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'christmas_tree.gif';
                        a.click();
                        URL.revokeObjectURL(url);
                        URL.revokeObjectURL(workerUrl);

                        btn.classList.remove('saving');
                        icon.innerHTML = '&#128190;';
                        icon.classList.remove('spinner');
                        text.textContent = 'Save GIF';
                        btn.disabled = false;
                        if (wasPlaying) togglePlay();
                    }});

                    const cameraState = camera.clone();
                    const nFrames = trajectoryData.length;
                    const animSpeed = 1000 / parseInt(document.getElementById('animation-speed-slider').value);

                    for (let f = 0; f < nFrames; f++) {{
                        updateScene(f, true);
                        camera.position.copy(cameraState.position);
                        camera.quaternion.copy(cameraState.quaternion);
                        camera.updateProjectionMatrix();
                        renderer.render(scene, camera);
                        gif.addFrame(renderer.domElement, {{ copy: true, delay: animSpeed }});
                        await new Promise(r => setTimeout(r, 10));
                    }}

                    gif.render();
                }} catch (error) {{
                    console.error("GIF Error:", error);
                    btn.classList.remove('saving');
                    icon.innerHTML = '&#9888;';
                    icon.classList.remove('spinner');
                    text.textContent = 'Error!';
                    btn.disabled = false;
                }}
            }}
            ========================================
            -->
            async function saveAsGif() {{
                const btn = document.getElementById('save-gif-btn');
                const icon = document.getElementById('gif-icon');
                const text = document.getElementById('gif-text');
                btn.classList.add('saving');
                icon.innerHTML = '⏳';
                icon.classList.add('spinner');
                text.textContent = 'Recording...';
                btn.disabled = true;
            
                try {{
                    const workerBlob = new Blob([gifWorkerCode], {{ type: 'application/javascript' }});
                    const workerUrl = URL.createObjectURL(workerBlob);
            
                    const targetFps = parseInt(document.getElementById('animation-speed-slider').value || 10);
                    const delayPerFrame = Math.round(100 / targetFps);  // gif.js uses hundredths of a second
            
                    const gif = new GIF({{
                        workers: 4,
                        quality: 10,
                        width: renderer.domElement.width,
                        height: renderer.domElement.height,
                        workerScript: workerUrl,
                        repeat: 0  // loop forever
                    }});
            
                    // Save current camera state to keep user view angle
                    const camPos = camera.position.clone();
                    const camQuat = camera.quaternion.clone();
                    controls.enabled = false;  // disable mouse control during recording
            
                    const totalFrames = trajectoryData.length;
            
                    for (let f = 0; f < totalFrames; f++) {{
                        // Update to current frame (may be slow - that's ok)
                        updateScene(f, true);
            
                        // Restore camera position and rotation
                        camera.position.copy(camPos);
                        camera.quaternion.copy(camQuat);
                        camera.updateProjectionMatrix();
            
                        // Force render
                        renderer.render(scene, camera);
            
                        // Add frame with fixed delay -> smooth GIF even if live playback lags
                        gif.addFrame(renderer.domElement, {{ copy: true, delay: delayPerFrame }});
            
                        // Give browser some breathing room to prevent freezing
                        await new Promise(r => requestAnimationFrame(() => setTimeout(r, 30)));
                    }}
            
                    gif.render();
            
                    gif.on('finished', function(blob) {{
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'christmas_tree_smooth.gif';
                        a.click();
            
                        URL.revokeObjectURL(url);
                        URL.revokeObjectURL(workerUrl);
            
                        // Restore controls
                        controls.enabled = true;
            
                        btn.classList.remove('saving');
                        icon.innerHTML = '📼';
                        icon.classList.remove('spinner');
                        text.textContent = 'Save GIF';
                        btn.disabled = false;
                    }});
            
                }} catch (error) {{
                    console.error("GIF Error:", error);
                    alert("GIF save failed: " + error.message);
                    btn.classList.remove('saving');
                    icon.innerHTML = '⚠️';
                    text.textContent = 'Error';
                    btn.disabled = false;
                }}
            }}

        </script>
    </body>
    </html>
    """

    if write_html:
        with open(write_html, 'w', encoding='utf-8') as f:
            f.write(html_template)

    return HTML(html_template)
