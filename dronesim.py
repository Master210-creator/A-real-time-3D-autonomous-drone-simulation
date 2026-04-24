import sys, math, random, time
import numpy as np
import pygame
from pygame.locals import *

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

if not OPENGL_AVAILABLE:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "PyOpenGL", "PyOpenGL_accelerate", "--break-system-packages", "-q"])
    from OpenGL.GL import *
    from OpenGL.GLU import *

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
W, H         = 1280, 720
WORLD_SIZE   = 40.0
GRID_CELLS   = 40
CELL_SIZE    = WORLD_SIZE / GRID_CELLS
NUM_TARGETS  = 8
SPOTLIGHT_R  = 5.5
DRONE_SPEED  = 0.08
DRONE_ALT    = 4.5
EDGE_THRESH  = 0.25

STREET_REGISTRY = {
    "Al Iqbal Road": (-12.0, -10.0),
    "Main Boulevard": (0.0, -15.0),
    "MM Alam Road": (10.0, -12.0),
    "Gulberg View": (15.0, 0.0),
    "Cavalry Ground": (-15.0, 0.0),
    "DHA Phase 1": (-10.0, 10.0),
    "Mall Road": (0.0, 15.0),
    "Ferozepur Road": (12.0, 10.0),
    "Canal Bank": (-16.0, -16.0),
    "Jail Road": (16.0, -16.0),
    "Liberty Market": (-16.0, 16.0),
    "Model Town": (16.0, 16.0)
}

COLORS = {
    "bg":        (0.01, 0.02, 0.02),
    "drone":     (0.40, 0.80, 1.00),
    "target":    (1.00, 0.30, 0.10),
    "found":     (0.10, 1.00, 0.40),
    "obstacle":  (0.25, 0.30, 0.35),
    "hud_text":  (0.70, 0.90, 1.00),
    "warn":      (1.00, 0.60, 0.00),
    "danger":    (1.00, 0.15, 0.15),
}

# ─────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────
def v3(x, y, z): return np.array([x, y, z], dtype=np.float32)
def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v
def lerp(a, b, t): return a + (b - a) * np.clip(t, 0, 1)
def lerp_angle(a, b, t):
    diff = (b - a + 180) % 360 - 180
    return a + diff * t

# ─────────────────────────────────────────────
#  OPENGL BOX GEOMETRY
# ─────────────────────────────────────────────
BOX_FACES = [
    ( 0, 0, 1, [(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]),
    ( 0, 0,-1, [(1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1)]),
    ( 0, 1, 0, [(-1,1,-1),(1,1,-1),(1,1,1),(-1,1,1)]),
    ( 0,-1, 0, [(-1,-1,1),(1,-1,1),(1,-1,-1),(-1,-1,-1)]),
    ( 1, 0, 0, [(1,-1,1),(1,-1,-1),(1,1,-1),(1,1,1)]),
    (-1, 0, 0, [(-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1)]),
]
_h = 0.5
BOX_EDGES = [
    ((-_h,-_h,-_h),(_h,-_h,-_h)),((_h,-_h,-_h),(_h,_h,-_h)),
    ((_h,_h,-_h),(-_h,_h,-_h)),((-_h,_h,-_h),(-_h,-_h,-_h)),
    ((-_h,-_h,_h),(_h,-_h,_h)),((_h,-_h,_h),(_h,_h,_h)),
    ((_h,_h,_h),(-_h,_h,_h)),((-_h,_h,_h),(-_h,-_h,_h)),
    ((-_h,-_h,-_h),(-_h,-_h,_h)),((_h,-_h,-_h),(_h,-_h,_h)),
    ((_h,_h,-_h),(_h,_h,_h)),((-_h,_h,-_h),(-_h,_h,_h)),
]

def draw_box():
    glBegin(GL_QUADS)
    for nx, ny, nz, verts in BOX_FACES:
        glNormal3f(nx, ny, nz)
        for vx, vy, vz in verts:
            glVertex3f(vx*0.5, vy*0.5, vz*0.5)
    glEnd()

def draw_box_wire():
    glBegin(GL_LINES)
    for v1, v2 in BOX_EDGES:
        glVertex3f(*v1); glVertex3f(*v2)
    glEnd()

def draw_cylinder(radius, height, slices=12):
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, height, slices, 1)
    glPushMatrix()
    glTranslatef(0, 0, height)
    gluDisk(gluNewQuadric(), 0, radius, slices, 1)
    glPopMatrix()
    glPushMatrix()
    glRotatef(180, 1, 0, 0)
    gluDisk(gluNewQuadric(), 0, radius, slices, 1)
    glPopMatrix()

def draw_cone(base_radius, height, slices=12):
    quad = gluNewQuadric()
    gluCylinder(quad, base_radius, 0.0, height, slices, 1)
    glPushMatrix()
    glRotatef(180, 1, 0, 0)
    gluDisk(gluNewQuadric(), 0, base_radius, slices, 1)
    glPopMatrix()

# ─────────────────────────────────────────────
#  MEMORY MAP
# ─────────────────────────────────────────────
class MemoryMap:
    def __init__(self, cells, bounding_box=None):
        self.cells   = cells
        self.bounding_box = bounding_box
        self.visited = np.zeros((cells, cells), dtype=np.float32)
        self.edge    = np.zeros((cells, cells), dtype=np.uint8)
        self.blocked = np.zeros((cells, cells), dtype=np.uint8)
        self.target_found = np.zeros((cells, cells), dtype=np.uint8)

    def world_to_cell(self, wx, wz):
        cx = int((wx + WORLD_SIZE/2) / CELL_SIZE)
        cz = int((wz + WORLD_SIZE/2) / CELL_SIZE)
        return (np.clip(cx, 0, self.cells-1), np.clip(cz, 0, self.cells-1))

    def cell_to_world(self, cx, cz):
        wx = cx * CELL_SIZE - WORLD_SIZE/2 + CELL_SIZE/2
        wz = cz * CELL_SIZE - WORLD_SIZE/2 + CELL_SIZE/2
        return wx, wz

    def illuminate(self, drone_x, drone_z, radius):
        r_cells = int(radius / CELL_SIZE) + 1
        cx0, cz0 = self.world_to_cell(drone_x, drone_z)
        changed = []
        for dx in range(-r_cells, r_cells+1):
            for dz in range(-r_cells, r_cells+1):
                cx, cz = cx0+dx, cz0+dz
                if not (0 <= cx < self.cells and 0 <= cz < self.cells): continue
                wx, wz = self.cell_to_world(cx, cz)
                dist = math.hypot(wx - drone_x, wz - drone_z)
                if dist <= radius:
                    intensity = max(0.3, 1.0 - (dist / radius)**2)
                    if self.visited[cx, cz] < intensity:
                        self.visited[cx, cz] = intensity
                        changed.append((cx, cz))
        for cx, cz in changed:
            if self.bounding_box:
                min_x, min_z, max_x, max_z = self.bounding_box
                wx, wz = self.cell_to_world(cx, cz)
                if not (min_x <= wx <= max_x and min_z <= wz <= max_z):
                    self.visited[cx, cz] = min(self.visited[cx, cz], 0.05)
            self._detect_edge(cx, cz)

    def _detect_edge(self, cx, cz):
        def v(x, z):
            if 0 <= x < self.cells and 0 <= z < self.cells:
                return float(self.visited[x, z])
            return 0.0
        gx = (v(cx+1,cz-1)+2*v(cx+1,cz)+v(cx+1,cz+1)
             -v(cx-1,cz-1)-2*v(cx-1,cz)-v(cx-1,cz+1))
        gz = (v(cx-1,cz+1)+2*v(cx,cz+1)+v(cx+1,cz+1)
             -v(cx-1,cz-1)-2*v(cx,cz-1)-v(cx+1,cz-1))
        mag = math.hypot(gx, gz) / 4.0
        self.edge[cx, cz] = 1 if mag > EDGE_THRESH else 0

    def frontier_cells(self):
        fron = []
        for cx in range(1, self.cells-1):
            for cz in range(1, self.cells-1):
                if self.visited[cx, cz] > 0 or self.blocked[cx, cz]: continue
                if self.bounding_box:
                    min_x, min_z, max_x, max_z = self.bounding_box
                    wx, wz = self.cell_to_world(cx, cz)
                    if not (min_x <= wx <= max_x and min_z <= wz <= max_z): continue
                for nx, nz in [(cx-1,cz),(cx+1,cz),(cx,cz-1),(cx,cz+1)]:
                    if self.visited[nx, nz] > 0:
                        fron.append((cx, cz)); break
        return fron

    def coverage(self):
        if self.bounding_box:
            min_x, min_z, max_x, max_z = self.bounding_box
            total = 0
            seen = 0
            for cx in range(self.cells):
                for cz in range(self.cells):
                    wx, wz = self.cell_to_world(cx, cz)
                    if min_x <= wx <= max_x and min_z <= wz <= max_z:
                        total += 1
                        if self.visited[cx, cz] > 0.05:
                            seen += 1
            if total == 0: return 0.0
            return seen / total * 100.0
        return np.count_nonzero(self.visited) / (self.cells * self.cells) * 100.0

# ─────────────────────────────────────────────
#  REALISTIC BUILDING  (Map 2 + urban night)
# ─────────────────────────────────────────────
class RealisticBuilding:
    # Palette: concrete, brick, glass-tower, tan sandstone
    STYLES = [
        {"wall": (0.45,0.45,0.48), "trim": (0.30,0.30,0.32), "roof": (0.25,0.25,0.28), "name": "concrete"},
        {"wall": (0.55,0.35,0.22), "trim": (0.38,0.22,0.12), "roof": (0.30,0.20,0.10), "name": "brick"},
        {"wall": (0.28,0.38,0.50), "trim": (0.18,0.28,0.42), "roof": (0.15,0.22,0.35), "name": "glass"},
        {"wall": (0.60,0.52,0.36), "trim": (0.42,0.36,0.22), "roof": (0.32,0.28,0.18), "name": "sandstone"},
    ]

    def __init__(self, x, z, w, d, h):
        self.x, self.z = x, z
        self.w, self.d, self.h = w, d, h
        self.style = random.choice(self.STYLES)
        # Windows: rows and cols
        self.win_cols = max(1, int(w / 1.2))
        self.win_rows = max(1, int(h / 1.5))
        self.win_lights = [
            [random.random() < 0.4 for _ in range(self.win_cols)]
            for _ in range(self.win_rows)
        ]
        self.has_rooftop = random.random() < 0.35
        self.antenna_h   = random.uniform(0.4, 1.2) if self.has_rooftop else 0

    def collides(self, px, pz, margin=0.6):
        return (abs(px - self.x) < self.w/2 + margin and
                abs(pz - self.z) < self.d/2 + margin)

    def draw(self, lit):
        s = self.style
        # Main body
        if lit:
            glColor4f(*s["wall"], 1.0)
        else:
            glColor4f(s["wall"][0]*0.08, s["wall"][1]*0.08, s["wall"][2]*0.10, 1.0)
        glPushMatrix()
        glTranslatef(self.x, self.h/2, self.z)
        glScalef(self.w, self.h, self.d)
        draw_box()
        glPopMatrix()

        if lit:
            # Trim edges
            glColor4f(*s["trim"], 0.9)
            glLineWidth(1.2)
            glPushMatrix()
            glTranslatef(self.x, self.h/2, self.z)
            glScalef(self.w+0.04, self.h+0.04, self.d+0.04)
            draw_box_wire()
            glPopMatrix()

            # Roof slab
            glColor4f(*s["roof"], 1.0)
            glPushMatrix()
            glTranslatef(self.x, self.h+0.05, self.z)
            glScalef(self.w+0.1, 0.12, self.d+0.1)
            draw_box()
            glPopMatrix()

            # Rooftop structure
            if self.has_rooftop:
                glColor4f(s["trim"][0]*0.8, s["trim"][1]*0.8, s["trim"][2]*0.8, 1.0)
                glPushMatrix()
                glTranslatef(self.x, self.h+0.05, self.z)
                glScalef(self.w*0.4, 0.4, self.d*0.4)
                draw_box()
                glPopMatrix()
                # Antenna
                glColor4f(0.6, 0.6, 0.65, 1.0)
                glPushMatrix()
                glTranslatef(self.x, self.h+0.45, self.z)
                glRotatef(-90, 1, 0, 0)
                draw_cylinder(0.04, self.antenna_h, 6)
                glPopMatrix()
                # Red blink light
                glColor4f(1.0, 0.1, 0.1, 0.9)
                glPushMatrix()
                glTranslatef(self.x, self.h+0.45+self.antenna_h, self.z)
                gluSphere(gluNewQuadric(), 0.07, 6, 6)
                glPopMatrix()

            # Windows
            self._draw_windows(lit)

    def _draw_windows(self, lit):
        if not lit: return
        face_offsets = [
            (0, 0, self.d/2+0.02, 0),    # front Z+
            (0, 0, -self.d/2-0.02, 180), # back  Z-
            (self.w/2+0.02, 0, 0, 90),   # right X+
            (-self.w/2-0.02, 0, 0, -90), # left  X-
        ]
        win_w = min(0.35, self.w / (self.win_cols+1) * 0.7)
        win_h = min(0.45, self.h / (self.win_rows+1) * 0.65)

        for ox, oy, oz, angle in face_offsets:
            face_w = self.d if abs(angle) == 90 else self.w
            for row in range(self.win_rows):
                for col in range(self.win_cols):
                    xf = (col - (self.win_cols-1)/2) * (face_w / (self.win_cols+1)) * 1.4
                    yf = 0.6 + row * (self.h / (self.win_rows+1)) * 1.0

                    on = self.win_lights[row][col]
                    if on:
                        glColor4f(1.0, 0.92, 0.65, 0.85)
                    else:
                        glColor4f(0.05, 0.08, 0.12, 0.7)

                    glPushMatrix()
                    glTranslatef(self.x+ox, oy+yf, self.z+oz)
                    glRotatef(angle, 0, 1, 0)
                    glScalef(win_w, win_h, 0.01)
                    draw_box()
                    glPopMatrix()

# ─────────────────────────────────────────────
#  TREE  (Forest Mode)
# ─────────────────────────────────────────────
class Tree:
    CANOPY_COLORS = [
        (0.08, 0.28, 0.08),
        (0.06, 0.22, 0.06),
        (0.10, 0.32, 0.10),
        (0.05, 0.18, 0.07),
        (0.12, 0.25, 0.08),
    ]
    TRUNK_COLORS = [
        (0.28, 0.18, 0.10),
        (0.22, 0.14, 0.08),
        (0.32, 0.20, 0.12),
    ]

    def __init__(self, x, z):
        self.x, self.z = x, z
        self.trunk_h  = random.uniform(1.8, 3.5)
        self.trunk_r  = random.uniform(0.10, 0.22)
        self.canopy_r = random.uniform(0.9, 2.0)
        self.canopy_h = random.uniform(2.0, 4.5)
        self.layers   = random.randint(2, 4)
        self.canopy_col = random.choice(self.CANOPY_COLORS)
        self.trunk_col  = random.choice(self.TRUNK_COLORS)
        self.sway_offset = random.uniform(0, math.pi*2)

    def collides(self, px, pz, margin=0.5):
        return math.hypot(px - self.x, pz - self.z) < self.canopy_r * 0.6 + margin

    def draw(self, lit, t):
        sway = math.sin(t * 0.4 + self.sway_offset) * 0.015

        # Trunk
        if lit:
            glColor4f(*self.trunk_col, 1.0)
        else:
            glColor4f(0.03, 0.02, 0.01, 1.0)
        glPushMatrix()
        glTranslatef(self.x, 0, self.z)
        glRotatef(-90, 1, 0, 0)
        draw_cylinder(self.trunk_r, self.trunk_h, 8)
        glPopMatrix()

        # Layered cones
        for i in range(self.layers):
            frac = i / max(1, self.layers - 1)
            layer_y = self.trunk_h + i * (self.canopy_h / self.layers) * 0.55
            layer_r = self.canopy_r * (1.0 - frac * 0.45)
            layer_h = self.canopy_h / self.layers * 1.3

            if lit:
                dark = 1.0 - frac * 0.3
                glColor4f(
                    self.canopy_col[0] * dark,
                    self.canopy_col[1] * dark,
                    self.canopy_col[2] * dark, 1.0
                )
            else:
                glColor4f(0.01, 0.03, 0.01, 1.0)

            glPushMatrix()
            glTranslatef(self.x + sway, layer_y, self.z + sway * 0.5)
            glRotatef(-90, 1, 0, 0)
            draw_cone(layer_r, layer_h, 10)
            glPopMatrix()

# ─────────────────────────────────────────────
#  FALLEN LOG  (Forest decoration obstacle)
# ─────────────────────────────────────────────
class FallenLog:
    def __init__(self, x, z):
        self.x, self.z = x, z
        self.angle = random.uniform(0, 360)
        self.length = random.uniform(2.0, 4.5)
        self.radius = random.uniform(0.12, 0.25)

    def collides(self, px, pz, margin=0.4):
        return math.hypot(px - self.x, pz - self.z) < self.length/2 + margin

    def draw(self, lit):
        if lit:
            glColor4f(0.30, 0.18, 0.10, 1.0)
        else:
            glColor4f(0.04, 0.02, 0.01, 1.0)
        glPushMatrix()
        glTranslatef(self.x, self.radius, self.z)
        glRotatef(self.angle, 0, 1, 0)
        glRotatef(90, 0, 1, 0)
        draw_cylinder(self.radius, self.length, 8)
        glPopMatrix()

# ─────────────────────────────────────────────
#  FIREFLY PARTICLE SYSTEM  (Forest only)
# ─────────────────────────────────────────────
class FireflySystem:
    def __init__(self, count=60):
        self.count = count
        self.pos   = np.random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2, (count, 3)).astype(np.float32)
        self.pos[:, 1] = np.random.uniform(0.3, 3.0, count)
        self.vel   = (np.random.rand(count, 3) - 0.5).astype(np.float32) * 0.012
        self.phase = np.random.uniform(0, math.pi*2, count)
        self.speed = np.random.uniform(0.3, 0.9, count)

    def update(self, dt):
        self.pos += self.vel
        # Bounce inside world
        half = WORLD_SIZE/2 - 1
        for i in range(3):
            lo = 0.2 if i == 1 else -half
            hi = 3.5 if i == 1 else  half
            mask_lo = self.pos[:, i] < lo
            mask_hi = self.pos[:, i] > hi
            self.vel[mask_lo | mask_hi, i] *= -1
        self.phase += dt * self.speed

    def draw(self):
        glDisable(GL_LIGHTING)
        glPointSize(2.5)
        glBegin(GL_POINTS)
        for i in range(self.count):
            brightness = (math.sin(self.phase[i]) + 1) / 2
            if brightness < 0.25: continue  # off phase
            glColor4f(0.7, 1.0, 0.3, brightness * 0.9)
            glVertex3f(*self.pos[i])
        glEnd()
        glEnable(GL_LIGHTING)

# ─────────────────────────────────────────────
#  RESCUE TARGET
# ─────────────────────────────────────────────
class Target:
    def __init__(self, x, z, target_type="rescue"):
        self.x, self.z   = x, z
        self.found       = False
        self.pulse       = random.uniform(0, math.pi*2)
        self.target_type = target_type

    def update(self, dt):
        self.pulse += dt * 3.0

    def draw(self, lit, t):
        if self.found and self.target_type != "dropoff":
            glColor4f(0.10, 1.00, 0.40, 0.9)
        elif lit:
            if self.target_type == "parcel":
                glColor4f(0.8, 0.4, 1.0, 0.95)
            elif self.target_type == "dropoff":
                glColor4f(0.2, 0.9, 0.8, 0.95)
            else:
                glColor4f(1.00, 0.30, 0.10, 0.95)
        else:
            if self.target_type == "dropoff":
                glColor4f(0.2, 0.9, 0.8, 0.3)
            else:
                return

        glPushMatrix()
        glTranslatef(self.x, 0.4, self.z)
        if not self.found:
            glScalef(1.0, 1.0+0.1*math.sin(self.pulse), 1.0)

        if self.target_type == "parcel":
            glScalef(0.4, 0.4, 0.4); draw_box()
        elif self.target_type == "dropoff":
            glPushMatrix()
            glTranslatef(0, -0.3, 0)
            glRotatef(90, 1, 0, 0)
            gluDisk(gluNewQuadric(), 1.0, 1.2, 20, 1)
            glPopMatrix()
            quad = gluNewQuadric()
            gluCylinder(quad, 0.1, 0.1, 1.0, 10, 2)
        else:
            gluSphere(gluNewQuadric(), 0.3, 10, 10)

        if lit and not self.found and self.target_type not in ("dropoff",):
            r = 0.3 + 0.4 * abs(math.sin(self.pulse * 0.5))
            glColor4f(1.0, 0.5, 0.1, max(0, 1.0 - r))
            glPushMatrix()
            glRotatef(90, 1, 0, 0)
            gluDisk(gluNewQuadric(), r, r+0.05, 20, 1)
            glPopMatrix()
        glPopMatrix()

# ─────────────────────────────────────────────
#  AI DRONE
# ─────────────────────────────────────────────
class AIDrone:
    STATE_EXPLORE  = "EXPLORE"
    STATE_APPROACH = "APPROACH"
    STATE_RESCUE   = "RESCUE"
    STATE_RETURN   = "RETURN"
    STATE_AVOID    = "AVOID"

    def __init__(self, memory, map_type=1):
        self.pos     = v3(0, DRONE_ALT, 0)
        self.vel     = v3(0, 0, 0)
        self.heading = 0.0
        self.memory  = memory
        self.map_type= map_type
        self.has_parcel = False
        self.state   = self.STATE_EXPLORE
        self.waypoint= None
        self.targets_found = 0
        self.total_targets = 0
        self.dist_traveled = 0.0
        self.rotor_spin    = 0.0
        self.avoid_timer   = 0.0
        self.last_pos      = self.pos.copy()
        self.battery       = 100.0
        self.mission_complete = False
        self._frontier_idx = 0
        self._stuck_timer  = 0.0
        self._scan_angle   = 0.0

    def set_waypoint(self, wx, wz):
        self.waypoint = v3(wx, DRONE_ALT, wz)

    def update(self, dt, obstacles, targets):
        if not self.mission_complete:
            self.battery = max(0.0, self.battery - (100.0/1200.0)*dt)
        self.rotor_spin = (self.rotor_spin + dt*400) % 360
        self._scan_angle += dt*60

        if self.map_type == 2:
            nearest_target = next((t for t in targets if t.target_type=="parcel" and not t.found), None) \
                if not self.has_parcel else \
                next((t for t in targets if t.target_type=="dropoff"), None)
        else:
            nearest_target = self._nearest_unrescued(targets)

        if self.battery < 15.0 and self.state not in (self.STATE_RETURN, self.STATE_AVOID):
            self.state = self.STATE_RETURN

        if self.state == self.STATE_RETURN:
            self.set_waypoint(0, 0)
            if math.hypot(self.pos[0], self.pos[2]) < 1.0:
                self.mission_complete = True
                self.waypoint = None

        elif self.state == self.STATE_EXPLORE:
            if nearest_target and self._distance2(nearest_target) < SPOTLIGHT_R*1.5:
                if nearest_target.target_type=="dropoff" and not self.has_parcel:
                    self._explore_step(dt)
                else:
                    self.state = self.STATE_APPROACH
                    self.set_waypoint(nearest_target.x, nearest_target.z)
            else:
                self._explore_step(dt)

        elif self.state == self.STATE_APPROACH:
            if nearest_target is None:
                self.state = self.STATE_EXPLORE
            elif self._distance2(nearest_target) < 1.2:
                self.state = self.STATE_RESCUE
            else:
                self.set_waypoint(nearest_target.x, nearest_target.z)

        elif self.state == self.STATE_RESCUE:
            if nearest_target and self._distance2(nearest_target) < 1.2:
                nearest_target.found = True
                if nearest_target.target_type == "parcel":
                    self.has_parcel = True
                elif nearest_target.target_type == "dropoff":
                    self.targets_found += 1
                    self.mission_complete = True
                else:
                    self.targets_found += 1
            self.state = self.STATE_EXPLORE

        elif self.state == self.STATE_AVOID:
            self.avoid_timer -= dt
            if self.avoid_timer <= 0:
                self.state = self.STATE_RETURN if self.battery < 15.0 else self.STATE_EXPLORE

        if self.waypoint is not None:
            diff = self.waypoint - self.pos
            diff[1] = 0
            dist = np.linalg.norm(diff)
            if dist > 0.3:
                desired = norm(diff) * DRONE_SPEED * 60 * dt
                repulse = self._repulsion(obstacles)
                desired += repulse
                self.vel = lerp(self.vel, desired, 0.15)
                self.pos += self.vel
                angle = math.degrees(math.atan2(diff[0], diff[2]))
                self.heading = lerp_angle(self.heading, angle, 0.08)
            else:
                self.vel *= 0.85
                if dist < 0.3: self.waypoint = None
        else:
            self.vel *= 0.85

        self.pos[1] = DRONE_ALT
        half = WORLD_SIZE/2 - 1.0
        self.pos[0] = np.clip(self.pos[0], -half, half)
        self.pos[2] = np.clip(self.pos[2], -half, half)

        step = np.linalg.norm(self.pos - self.last_pos)
        self.dist_traveled += step
        self.last_pos = self.pos.copy()

        if step < 0.001 and not self.mission_complete:
            self._stuck_timer += dt
            if self._stuck_timer > 2.0:
                self._stuck_timer = 0
                self.state = self.STATE_AVOID
                self.avoid_timer = 1.0
                angle = math.radians(self.heading) + math.pi + random.uniform(-0.5, 0.5)
                self.set_waypoint(self.pos[0]+3*math.cos(angle), self.pos[2]+3*math.sin(angle))
        else:
            self._stuck_timer = 0

        self.memory.illuminate(self.pos[0], self.pos[2], SPOTLIGHT_R)

    def _explore_step(self, dt):
        if self.waypoint is None:
            frontiers = self.memory.frontier_cells()
            if frontiers:
                best = min(frontiers, key=lambda c: math.hypot(
                    self.memory.cell_to_world(*c)[0]-self.pos[0],
                    self.memory.cell_to_world(*c)[1]-self.pos[2]))
                wx, wz = self.memory.cell_to_world(*best)
                self.set_waypoint(wx, wz)
            else:
                angle = math.radians(self._scan_angle)
                r = random.uniform(3, 10)
                self.set_waypoint(math.cos(angle)*r, math.sin(angle)*r)

    def _repulsion(self, obstacles):
        rep = v3(0,0,0)
        for obs in obstacles:
            ox = obs.x if hasattr(obs,'x') else obs[0]
            oz = obs.z if hasattr(obs,'z') else obs[1]
            dx = self.pos[0]-ox; dz = self.pos[2]-oz
            dist = math.hypot(dx, dz)
            if dist < 3.0 and dist > 0.01:
                strength = (3.0-dist)/3.0*0.15
                rep += v3(dx/dist*strength, 0, dz/dist*strength)
        return rep

    def _nearest_unrescued(self, targets):
        best, bd = None, 1e9
        for t in targets:
            if not t.found:
                d = self._distance2(t)
                if d < bd: bd, best = d, t
        return best

    def _distance2(self, target):
        return math.hypot(self.pos[0]-target.x, self.pos[2]-target.z)

    def draw(self, t):
        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(-self.heading, 0, 1, 0)
        glColor4f(*COLORS["drone"], 1.0)
        gluSphere(gluNewQuadric(), 0.35, 12, 8)
        if getattr(self,'has_parcel',False):
            glColor4f(0.8, 0.4, 1.0, 1.0)
            glPushMatrix()
            glTranslatef(0, -0.4, 0)
            glScalef(0.4, 0.4, 0.4)
            draw_box()
            glPopMatrix()
        for i in range(4):
            angle = 45 + i*90
            ax = math.cos(math.radians(angle))*0.7
            az = math.sin(math.radians(angle))*0.7
            glColor4f(0.3, 0.4, 0.5, 1.0)
            glPushMatrix()
            glTranslatef(ax*0.5, 0, az*0.5)
            glRotatef(angle, 0, 1, 0)
            glScalef(0.7, 0.05, 0.08)
            draw_box()
            glPopMatrix()
            glColor4f(0.5, 0.8, 1.0, 0.5)
            glPushMatrix()
            glTranslatef(ax, 0.1, az)
            glRotatef(self.rotor_spin+i*90, 0, 1, 0)
            glScalef(0.4, 0.01, 0.4)
            draw_box()
            glPopMatrix()
        glColor4f(1.0, 0.95, 0.7, 0.07)
        glPushMatrix()
        glRotatef(90, 1, 0, 0)
        gluCylinder(gluNewQuadric(), 0.1, SPOTLIGHT_R, DRONE_ALT, 20, 2)
        glPopMatrix()
        glPopMatrix()

# ─────────────────────────────────────────────
#  GROUND RENDERING
# ─────────────────────────────────────────────
def draw_ground(size, map_type, scene_variant="urban"):
    glDisable(GL_LIGHTING)
    half = size/2
    cell = size/GRID_CELLS

    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            x0 = -half + i*cell
            z0 = -half + j*cell
            checker = (i+j) % 2

            if scene_variant == "forest":
                # Dark mossy forest floor
                base = 0.07 + checker*0.02
                r_var = random.uniform(0.95, 1.05) if random.random() < 0.05 else 1.0
                glColor4f(base*0.7*r_var, base*1.2*r_var, base*0.5*r_var, 1.0)
            elif map_type in (2, 3):
                shade = 0.50 + checker*0.05
                glColor4f(shade*0.75, shade*0.85, shade*0.72, 1.0)
            else:
                shade = 0.04 + checker*0.02
                glColor4f(shade, shade*1.2, shade*1.5, 1.0)

            glBegin(GL_QUADS)
            glVertex3f(x0,      0, z0)
            glVertex3f(x0+cell, 0, z0)
            glVertex3f(x0+cell, 0, z0+cell)
            glVertex3f(x0,      0, z0+cell)
            glEnd()

    # Grid lines
    if scene_variant == "forest":
        glColor4f(0.05, 0.10, 0.04, 0.3)
    elif map_type in (2, 3):
        glColor4f(0.3, 0.4, 0.3, 0.4)
    else:
        glColor4f(0.08, 0.12, 0.18, 0.5)

    glLineWidth(0.5)
    glBegin(GL_LINES)
    for i in range(GRID_CELLS+1):
        x = -half + i*cell
        glVertex3f(x, 0.01, -half); glVertex3f(x, 0.01, half)
        glVertex3f(-half, 0.01, x); glVertex3f(half, 0.01, x)
    glEnd()

    # Forest: scatter leaf/moss patches
    if scene_variant == "forest":
        random.seed(999)
        glBegin(GL_QUADS)
        for _ in range(180):
            px = random.uniform(-half+0.5, half-0.5)
            pz = random.uniform(-half+0.5, half-0.5)
            s  = random.uniform(0.15, 0.55)
            g  = random.uniform(0.06, 0.14)
            glColor4f(g*0.6, g*1.3, g*0.4, 0.8)
            glVertex3f(px-s, 0.02, pz-s)
            glVertex3f(px+s, 0.02, pz-s)
            glVertex3f(px+s, 0.02, pz+s)
            glVertex3f(px-s, 0.02, pz+s)
        glEnd()
        random.seed()

    glEnable(GL_LIGHTING)

# ─────────────────────────────────────────────
#  SPOTLIGHT POOL
# ─────────────────────────────────────────────
def draw_spotlight_pool(drone_x, drone_z, map_type, scene_variant="urban"):
    if map_type in (2, 3) and scene_variant == "urban": return
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    steps = 60
    for ring in range(5, 0, -1):
        r = SPOTLIGHT_R * ring/5
        alpha = 0.06 * (6-ring)
        glColor4f(1.0, 0.95, 0.8, alpha)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(drone_x, 0.02, drone_z)
        for i in range(steps+1):
            a = i/steps*math.pi*2
            glVertex3f(drone_x+math.cos(a)*r, 0.02, drone_z+math.sin(a)*r)
        glEnd()
    glEnable(GL_LIGHTING)

# ─────────────────────────────────────────────
#  STARS
# ─────────────────────────────────────────────
def draw_stars(map_type, scene_variant):
    if map_type in (2,3) and scene_variant == "urban": return
    glDisable(GL_LIGHTING)
    glPointSize(1.5)
    glBegin(GL_POINTS)
    random.seed(42)
    for _ in range(300):
        a  = random.uniform(0, math.pi*2)
        el = random.uniform(0.1, 0.6)
        r  = 200
        x  = r*math.cos(el)*math.cos(a)
        y  = r*math.sin(el)
        z  = r*math.cos(el)*math.sin(a)
        br = random.uniform(0.4, 1.0)
        glColor4f(br, br, br*0.95, 1.0)
        glVertex3f(x, y, z)
    glEnd()
    random.seed()
    glEnable(GL_LIGHTING)

# ─────────────────────────────────────────────
#  LIGHTING SETUP
# ─────────────────────────────────────────────
def setup_lighting(drone_pos, map_type, scene_variant="urban"):
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    if map_type in (2,3) and scene_variant == "urban":
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 50.0, 10.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.95, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180.0)
    else:
        # Night / forest: deep ambient darkness, drone spotlight
        if scene_variant == "forest":
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.005, 0.012, 0.005, 1.0])
        else:
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.02, 0.03, 0.04, 1.0])
        pos = [drone_pos[0], drone_pos[1], drone_pos[2], 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, pos)
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.95, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, [0.0, -1.0, 0.0])
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF,    50.0)
        glLightf(GL_LIGHT0, GL_SPOT_EXPONENT,   8.0)
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION,  0.1)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION,    0.05)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.02)

# ─────────────────────────────────────────────
#  SCENE GENERATION
# ─────────────────────────────────────────────
def generate_scene(memory, map_type=1, scene_variant="urban", delivery_coords=None):
    obstacles = []  # generic list (buildings OR trees+logs)
    targets   = []
    placed    = []  # (x, z, radius) for collision checks

    def place_ok(x, z, r):
        if math.hypot(x, z) < 3.5: return False
        for ox, oz, or_ in placed:
            if math.hypot(x-ox, z-oz) < r+or_+0.8: return False
        return True

    # ── FOREST MODE ────────────────────────
    if scene_variant == "forest":
        # Dense tree clusters
        num_trees = 55
        for _ in range(num_trees):
            for attempt in range(60):
                x = random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2)
                z = random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2)
                r = random.uniform(0.9, 2.0)
                if place_ok(x, z, r):
                    tree = Tree(x, z)
                    obstacles.append(tree)
                    placed.append((x, z, tree.canopy_r * 0.6))
                    cx, cz = memory.world_to_cell(x, z)
                    memory.blocked[cx, cz] = 1
                    break

        # Fallen logs (impassable debris)
        for _ in range(12):
            for attempt in range(40):
                x = random.uniform(-WORLD_SIZE/2+3, WORLD_SIZE/2-3)
                z = random.uniform(-WORLD_SIZE/2+3, WORLD_SIZE/2-3)
                if place_ok(x, z, 2.5):
                    log = FallenLog(x, z)
                    obstacles.append(log)
                    placed.append((x, z, log.length/2))
                    break

        # Survivors hidden in forest
        for _ in range(NUM_TARGETS):
            for attempt in range(100):
                x = random.uniform(-WORLD_SIZE/2+1, WORLD_SIZE/2-1)
                z = random.uniform(-WORLD_SIZE/2+1, WORLD_SIZE/2-1)
                if math.hypot(x, z) < 2.0: continue
                clear = not any(
                    math.hypot(x-ox, z-oz) < or_+1.2
                    for ox, oz, or_ in placed
                )
                if clear:
                    targets.append(Target(x, z, "rescue"))
                    break

    # ── URBAN NIGHT (Map 1 original, but with realistic buildings) ──
    elif map_type == 1:
        for _ in range(18):
            for attempt in range(40):
                x = random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2)
                z = random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2)
                w = random.uniform(2.0, 5.0)
                d = random.uniform(2.0, 5.0)
                h = random.uniform(2.0, 8.0)
                r = math.hypot(w/2, d/2)
                if place_ok(x, z, r):
                    bld = RealisticBuilding(x, z, w, d, h)
                    obstacles.append(bld)
                    placed.append((x, z, r))
                    cx, cz = memory.world_to_cell(x, z)
                    memory.blocked[cx, cz] = 1
                    break
        for _ in range(NUM_TARGETS):
            for attempt in range(100):
                x = random.uniform(-WORLD_SIZE/2+1, WORLD_SIZE/2-1)
                z = random.uniform(-WORLD_SIZE/2+1, WORLD_SIZE/2-1)
                if math.hypot(x, z) < 2.0: continue
                clear = not any(
                    math.hypot(x-ox, z-oz) < or_+1.5
                    for ox, oz, or_ in placed
                )
                if clear:
                    targets.append(Target(x, z, "rescue"))
                    break

    # ── DAYLIGHT DELIVERY (Map 2) ──
    elif map_type == 2:
        hq = RealisticBuilding(0, 0, 6.0, 6.0, 10.0)
        obstacles.append(hq)
        placed.append((0, 0, 4.5))
        memory.blocked[memory.world_to_cell(0,0)[0], memory.world_to_cell(0,0)[1]] = 1

        if delivery_coords:
            p_pos, d_pos = delivery_coords
            targets.append(Target(p_pos[0], p_pos[1], "parcel"))
            targets.append(Target(d_pos[0], d_pos[1], "dropoff"))
            placed.append((p_pos[0], p_pos[1], 1.5))
            placed.append((d_pos[0], d_pos[1], 1.5))

        for _ in range(18):
            for attempt in range(40):
                x = random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2)
                z = random.uniform(-WORLD_SIZE/2+2, WORLD_SIZE/2-2)
                w = random.uniform(1.5, 4.5)
                d = random.uniform(1.5, 4.5)
                h = random.uniform(1.5, 7.0)
                r = math.hypot(w/2, d/2)
                if place_ok(x, z, r):
                    bld = RealisticBuilding(x, z, w, d, h)
                    obstacles.append(bld)
                    placed.append((x, z, r))
                    cx, cz = memory.world_to_cell(x, z)
                    memory.blocked[cx, cz] = 1
                    break

        dropoff_x, dropoff_z = 0, 6.0
        targets.append(Target(dropoff_x, dropoff_z, "dropoff"))
        for attempt in range(100):
            x = random.uniform(-WORLD_SIZE/2+4, WORLD_SIZE/2-4)
            z = random.uniform(-WORLD_SIZE/2+4, WORLD_SIZE/2-4)
            if math.hypot(x, z) < 8.0: continue
            clear = not any(math.hypot(x-ox, z-oz) < or_+1.5 for ox,oz,or_ in placed)
            if clear:
                targets.append(Target(x, z, "parcel"))
                break

    # ── OBSTACLE AVOIDANCE (Map 3) ──
    elif map_type == 3:
        cell_size = 4.0
        grid_w = int(WORLD_SIZE/cell_size)
        for i in range(grid_w):
            for j in range(grid_w):
                # Increased density and removed grid restrictions for a true obstacle course
                if random.random()<0.65 and (1<i<grid_w-2) and (1<j<grid_w-2):
                    x = -WORLD_SIZE/2 + i*cell_size + cell_size/2 + random.uniform(-1, 1)
                    z = -WORLD_SIZE/2 + j*cell_size + cell_size/2 + random.uniform(-1, 1)
                    if math.hypot(x,z)>3.5:
                        w = random.uniform(1.5, 2.5)
                        d = random.uniform(1.5, 2.5)
                        h = random.uniform(3.0, 10.0)
                        obstacles.append(RealisticBuilding(x, z, w, d, h))
                        cx, cz = memory.world_to_cell(x, z)
                        memory.blocked[cx, cz] = 1

    return obstacles, targets

# ─────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────
class HUD:
    def __init__(self):
        pygame.font.init()
        self.font_big = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_med = pygame.font.SysFont("Consolas", 16)
        self.font_sm  = pygame.font.SysFont("Consolas", 13)
        self.surf     = pygame.Surface((W, H), pygame.SRCALPHA)

    def draw(self, screen, drone, memory, total_targets, elapsed, paused, scene_variant="urban"):
        self.surf.fill((0,0,0,0))
        s = self.surf

        # Mini-map
        mm_size = 170
        mm_x, mm_y = W-mm_size-12, 10
        pygame.draw.rect(s, (5,12,20,220), (mm_x-2, mm_y-2, mm_size+4, mm_size+4))
        pygame.draw.rect(s, (20,60,100,180), (mm_x, mm_y, mm_size, mm_size))
        cell_px = mm_size/GRID_CELLS
        for cx in range(GRID_CELLS):
            for cz in range(GRID_CELLS):
                v = memory.visited[cx,cz]
                if v > 0:
                    if scene_variant == "forest":
                        col = (int(10+v*20), int(40+v*120), int(10+v*30), 200)
                    else:
                        col = (int(20+v*30), int(50+v*100), int(80+v*120), 200)
                    pygame.draw.rect(s, col,
                        (mm_x+cx*cell_px, mm_y+cz*cell_px,
                         max(1,cell_px-0.5), max(1,cell_px-0.5)))
                if memory.edge[cx,cz]:
                    pygame.draw.rect(s, (0,200,255,180),
                        (mm_x+cx*cell_px, mm_y+cz*cell_px, max(1,cell_px), max(1,cell_px)))
                if memory.target_found[cx,cz]:
                    pygame.draw.rect(s, (50,255,100,220),
                        (mm_x+cx*cell_px, mm_y+cz*cell_px, max(1,cell_px), max(1,cell_px)))

        dx_mm = mm_x + (drone.pos[0]/WORLD_SIZE+0.5)*mm_size
        dz_mm = mm_y + (drone.pos[2]/WORLD_SIZE+0.5)*mm_size
        sl_r  = SPOTLIGHT_R/WORLD_SIZE*mm_size
        pygame.draw.circle(s, (255,240,150,40), (int(dx_mm),int(dz_mm)), int(sl_r))
        pygame.draw.circle(s, (100,200,255,255), (int(dx_mm),int(dz_mm)), 4)

        mode_lbl = "FOREST MAP" if scene_variant=="forest" else "MEMORY MAP"
        label = self.font_sm.render(mode_lbl, True, (100,220,140) if scene_variant=="forest" else (100,180,220))
        s.blit(label, (mm_x, mm_y+mm_size+3))

        # Status panel
        panel_x, panel_y = 10, H-190-10
        panel_w, panel_h = 430, 190
        pygame.draw.rect(s, (3,8,15,210), (panel_x,panel_y,panel_w,panel_h), border_radius=8)
        pygame.draw.rect(s, (20,60,100,100), (panel_x,panel_y,panel_w,panel_h), width=1, border_radius=8)

        def text(txt, x, y, font=None, color=(180,220,255)):
            f = font or self.font_med
            surf = f.render(txt, True, color)
            s.blit(surf, (panel_x+x, panel_y+y))

        if scene_variant == "forest":
            text("NIGHT RESCUE - FOREST MODE", 10, 8, self.font_big, (100,255,140))
        else:
            text("NIGHT RESCUE DRONE", 10, 8, self.font_big, (100,210,255))

        mins = int(elapsed//60); secs = int(elapsed%60)
        text(f"TIME   {mins:02d}:{secs:02d}", 10, 38)
        text(f"STATE  {drone.state}", 10, 58,
             color=(100,255,150) if drone.state=="RESCUE" else (180,220,255))
        text(f"FOUND  {drone.targets_found}/{total_targets}", 10, 78,
             color=(100,255,100) if drone.targets_found==total_targets else (255,200,80))
        text(f"DIST   {drone.dist_traveled:.1f}m", 10, 98)
        text(f"MAP    {memory.coverage():.1f}%", 10, 118)

        b_color = (100,255,100)
        if drone.battery < 30: b_color = (255,200,80)
        if drone.battery < 15: b_color = (255,50,50)
        text(f"BATT   {drone.battery:.1f}%", 10, 138, color=b_color)

        bar_x, bar_y, bar_w = 200, 118, 180
        cov = memory.coverage()/100.0
        pygame.draw.rect(s, (20,40,60), (panel_x+bar_x, panel_y+bar_y+4, bar_w, 10), border_radius=4)
        bar_color = (0,180,60) if scene_variant=="forest" else (0,200,100)
        pygame.draw.rect(s, bar_color, (panel_x+bar_x, panel_y+bar_y+4, int(bar_w*cov), 10), border_radius=4)

        text(f"POS    ({drone.pos[0]:+.1f}, {drone.pos[2]:+.1f})", 10, 158)
        if drone.mission_complete:
            text("MISSION COMPLETE!", 200, 158, color=(100,255,100))

        spd = np.linalg.norm(drone.vel)*60
        text(f"SPD  {spd:.2f} m/s", 265, 38)
        text(f"HDG  {drone.heading%360:.0f}°", 265, 58)
        text(f"ALT  {drone.pos[1]:.1f} m", 265, 78)
        text(f"EDGES  {int(np.sum(memory.edge))}", 265, 98)

        # Controls
        hints_x = W-mm_size-12
        hints_y = mm_y+mm_size+22
        hints = ["SPACE  Pause/Resume", "R      Reset", "C      Toggle camera", "ESC    Quit"]
        for i, h in enumerate(hints):
            hs = self.font_sm.render(h, True, (60,120,160))
            s.blit(hs, (hints_x, hints_y+i*17))

        if paused:
            ps = self.font_big.render("PAUSED", True, (255,200,50))
            s.blit(ps, (W//2-ps.get_width()//2, H//2))

        if drone.targets_found == total_targets and total_targets > 0:
            vs = self.font_big.render("ALL SURVIVORS RESCUED!", True, (50,255,100))
            s.blit(vs, (W//2-vs.get_width()//2, H//2-30))

        screen.blit(s, (0,0))

# ─────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.mode  = 0
        self.yaw   = 30.0
        self.pitch = 35.0
        self.dist  = 20.0
        self._target = v3(0,0,0)

    def apply(self, drone_pos, dt):
        self._target = lerp(self._target, drone_pos, 0.05)
        tx, ty, tz = self._target
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if self.mode == 0:
            offset_angle = math.radians(self.yaw)
            cx = tx + math.sin(offset_angle)*self.dist
            cy = ty + self.dist*math.tan(math.radians(self.pitch))
            cz = tz + math.cos(offset_angle)*self.dist
            gluLookAt(cx, cy, cz, tx, ty, tz, 0, 1, 0)
        elif self.mode == 1:
            gluLookAt(tx, 35, tz, tx, 0, tz, 0, 0, -1)
        elif self.mode == 2:
            cx = self.dist*math.sin(math.radians(self.yaw))*math.cos(math.radians(self.pitch))
            cy = self.dist*math.sin(math.radians(self.pitch))
            cz = self.dist*math.cos(math.radians(self.yaw))*math.cos(math.radians(self.pitch))
            gluLookAt(cx, cy, cz, 0, 0, 0, 0, 1, 0)

# ─────────────────────────────────────────────
#  ADDRESS ENTRY SCREEN (MAP 2)
# ─────────────────────────────────────────────
class AddressEntryScreen:
    def __init__(self, screen):
        self.screen = screen
        self.addresses = list(STREET_REGISTRY.keys())
        self.pickup_idx = 0
        self.dest_idx = 1
        self.active_field = 0 # 0 for pickup, 1 for dest
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Consolas", 34, bold=True)
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_sm = pygame.font.SysFont("Consolas", 17)

    def run(self):
        running = True
        error_msg = ""
        while running:
            self.screen.fill((13, 27, 42)) # Dark background

            title = self.font_title.render("CONFIGURE DELIVERY MISSION", True, (26, 115, 232))
            self.screen.blit(title, (W//2-title.get_width()//2, 50))
            
            sub = self.font_sm.render("Step 1 of 1: Select Pickup and Destination Addresses", True, (160, 180, 200))
            self.screen.blit(sub, (W//2-sub.get_width()//2, 100))

            # Fields
            for i, label in enumerate(["Pickup Address", "Destination Address"]):
                y = 200 + i * 100
                color = (255, 255, 255) if self.active_field == i else (150, 150, 150)
                box_color = (40, 80, 120) if self.active_field == i else (20, 40, 60)
                
                lbl_surf = self.font_sm.render(label, True, color)
                self.screen.blit(lbl_surf, (W//4 - 50, y - 25))
                
                pygame.draw.rect(self.screen, box_color, (W//4 - 50, y, 400, 50), border_radius=5)
                val = self.addresses[self.pickup_idx] if i == 0 else self.addresses[self.dest_idx]
                val_surf = self.font.render(val, True, (220, 230, 240))
                self.screen.blit(val_surf, (W//4 - 35, y + 10))

            if error_msg:
                err = self.font_sm.render(error_msg, True, (255, 100, 100))
                self.screen.blit(err, (W//4 - 50, 400))

            # Minimap Preview
            p_pos = STREET_REGISTRY[self.addresses[self.pickup_idx]]
            d_pos = STREET_REGISTRY[self.addresses[self.dest_idx]]
            self._draw_minimap_preview(p_pos, d_pos)

            hint = self.font_sm.render("Use UP/DOWN to select | TAB to switch | ENTER to launch | ESC to cancel", True, (100, 130, 160))
            self.screen.blit(hint, (W//2-hint.get_width()//2, 650))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return None, None
                    elif event.key in (K_UP, K_DOWN):
                        delta = -1 if event.key == K_UP else 1
                        if self.active_field == 0:
                            self.pickup_idx = (self.pickup_idx + delta) % len(self.addresses)
                        else:
                            self.dest_idx = (self.dest_idx + delta) % len(self.addresses)
                    elif event.key == K_TAB:
                        self.active_field = 1 - self.active_field
                    elif event.key == K_RETURN:
                        if self.pickup_idx == self.dest_idx:
                            error_msg = "Pickup and Destination cannot be identical."
                        else:
                            return self.addresses[self.pickup_idx], self.addresses[self.dest_idx]
        return None, None

    def _draw_minimap_preview(self, p_pos, d_pos):
        mm_size = 300
        mm_x, mm_y = W//2 + 100, 150
        pygame.draw.rect(self.screen, (20, 30, 40), (mm_x, mm_y, mm_size, mm_size))
        pygame.draw.rect(self.screen, (50, 80, 120), (mm_x, mm_y, mm_size, mm_size), width=2)
        
        # Grid lines
        for i in range(10):
            pygame.draw.line(self.screen, (40, 50, 60), (mm_x, mm_y + i*mm_size/10), (mm_x+mm_size, mm_y + i*mm_size/10))
            pygame.draw.line(self.screen, (40, 50, 60), (mm_x + i*mm_size/10, mm_y), (mm_x + i*mm_size/10, mm_y+mm_size))

        def world_to_mm(wx, wz):
            x = mm_x + (wx / WORLD_SIZE + 0.5) * mm_size
            y = mm_y + (wz / WORLD_SIZE + 0.5) * mm_size
            return int(x), int(y)

        p_px = world_to_mm(*p_pos)
        d_px = world_to_mm(*d_pos)

        # Distance line
        pygame.draw.line(self.screen, (255, 255, 255), p_px, d_px, 2)
        dist = math.hypot(p_pos[0]-d_pos[0], p_pos[1]-d_pos[1])
        dist_lbl = self.font_sm.render(f"Dist: {dist:.1f}m", True, (200, 200, 200))
        self.screen.blit(dist_lbl, (mm_x, mm_y + mm_size + 10))

        pygame.draw.circle(self.screen, (200, 100, 255), p_px, 8) # Pickup
        pygame.draw.circle(self.screen, (50, 255, 200), d_px, 8)  # Dest
        
        lbl_p = self.font_sm.render("P", True, (0, 0, 0))
        self.screen.blit(lbl_p, (p_px[0]-4, p_px[1]-6))
        lbl_d = self.font_sm.render("D", True, (0, 0, 0))
        self.screen.blit(lbl_d, (d_px[0]-4, d_px[1]-6))


# ─────────────────────────────────────────────
#  AREA DRAWING SCREEN (MAP 3)
# ─────────────────────────────────────────────
class AreaDrawingScreen:
    def __init__(self, screen, memory):
        self.screen = screen
        self.memory = memory
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Consolas", 34, bold=True)
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_sm = pygame.font.SysFont("Consolas", 17)
        self.mm_size = 500
        self.mm_x = W//2 - self.mm_size//2
        self.mm_y = 120
        self.start_pos = None
        self.end_pos = None
        self.rect = None # (min_x, min_z, max_x, max_z) in world

    def run(self):
        running = True
        mouse_down = False
        error_msg = ""
        while running:
            self.screen.fill((13, 27, 42))

            title = self.font_title.render("DEFINE SEARCH AREA", True, (26, 115, 232))
            self.screen.blit(title, (W//2-title.get_width()//2, 30))
            sub = self.font_sm.render("Draw a rectangle to restrict drone exploration", True, (160, 180, 200))
            self.screen.blit(sub, (W//2-sub.get_width()//2, 70))

            self._draw_grid_with_obstacles()

            if self.start_pos and self.end_pos:
                px = min(self.start_pos[0], self.end_pos[0])
                py = min(self.start_pos[1], self.end_pos[1])
                pw = abs(self.start_pos[0] - self.end_pos[0])
                ph = abs(self.start_pos[1] - self.end_pos[1])
                
                # Highlight
                s = pygame.Surface((pw, ph), pygame.SRCALPHA)
                s.fill((50, 150, 255, 80))
                self.screen.blit(s, (px, py))
                pygame.draw.rect(self.screen, (100, 200, 255), (px, py, pw, ph), width=2)
                
                wx1, wz1 = self._screen_to_world(px, py)
                wx2, wz2 = self._screen_to_world(px+pw, py+ph)
                self.rect = (wx1, wz1, wx2, wz2)
                
                w_world = abs(wx2 - wx1)
                h_world = abs(wz2 - wz1)
                info = self.font.render(f"Area: {w_world:.1f} x {h_world:.1f} m  ({w_world*h_world:.1f} sq m)", True, (200, 255, 200))
                self.screen.blit(info, (W//2-info.get_width()//2, self.mm_y + self.mm_size + 20))
                
                if w_world < 4.0 or h_world < 4.0:
                    error_msg = "Area too small! Minimum size is 4x4 meters."
                else:
                    error_msg = ""
            
            if error_msg:
                err = self.font_sm.render(error_msg, True, (255, 100, 100))
                self.screen.blit(err, (W//2-err.get_width()//2, self.mm_y + self.mm_size + 50))

            hint = self.font_sm.render("Drag Left Mouse to Draw | Right Click to Clear | ENTER to Launch | ESC to Cancel", True, (100, 130, 160))
            self.screen.blit(hint, (W//2-hint.get_width()//2, H - 40))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return None
                    elif event.key == K_RETURN:
                        if self.rect:
                            w_world = self.rect[2] - self.rect[0]
                            h_world = self.rect[3] - self.rect[1]
                            if w_world >= 4.0 and h_world >= 4.0:
                                return self.rect
                    elif event.key == K_r:
                        self.start_pos = None; self.end_pos = None; self.rect = None
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.start_pos = event.pos
                        self.end_pos = event.pos
                        mouse_down = True
                    elif event.button == 3:
                        self.start_pos = None; self.end_pos = None; self.rect = None
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        mouse_down = False
                elif event.type == MOUSEMOTION:
                    if mouse_down:
                        # constrain to map
                        ex = np.clip(event.pos[0], self.mm_x, self.mm_x + self.mm_size)
                        ey = np.clip(event.pos[1], self.mm_y, self.mm_y + self.mm_size)
                        self.end_pos = (ex, ey)
        return None

    def _screen_to_world(self, px, py):
        nx = (px - self.mm_x) / self.mm_size
        ny = (py - self.mm_y) / self.mm_size
        wx = (nx - 0.5) * WORLD_SIZE
        wz = (ny - 0.5) * WORLD_SIZE
        return wx, wz

    def _draw_grid_with_obstacles(self):
        pygame.draw.rect(self.screen, (20, 30, 40), (self.mm_x, self.mm_y, self.mm_size, self.mm_size))
        cell_px = self.mm_size / GRID_CELLS
        for cx in range(GRID_CELLS):
            for cz in range(GRID_CELLS):
                if self.memory.blocked[cx, cz]:
                    px = self.mm_x + cx * cell_px
                    py = self.mm_y + cz * cell_px
                    pygame.draw.rect(self.screen, (80, 80, 90), (px, py, cell_px+1, cell_px+1))


# ─────────────────────────────────────────────
#  MAP SELECTION MENU
# ─────────────────────────────────────────────
def select_map_menu(screen):
    font_title = pygame.font.SysFont("Consolas", 34, bold=True)
    font       = pygame.font.SysFont("Consolas", 24, bold=True)
    font_sm    = pygame.font.SysFont("Consolas", 17)

    options = [
        ("[1]", "NIGHT RESCUE — URBAN",         "(Original city mode)",      (100,180,255)),
        ("[2]", "NIGHT RESCUE — FOREST",         "(New! Dense forest + fireflies)", (100,255,140)),
        ("[3]", "DAYLIGHT DELIVERY",             "(Logistics / parcel drop)", (255,200,80)),
    ]

    running = True
    while running:
        screen.fill((8, 12, 18))

        title = font_title.render("SELECT MISSION", True, (100,210,255))
        screen.blit(title, (W//2-title.get_width()//2, 150))

        sub = font_sm.render("Autonomous Drone Simulation  v2.0", True, (60,100,130))
        screen.blit(sub, (W//2-sub.get_width()//2, 195))

        for i, (key, name, desc, col) in enumerate(options):
            y = 270 + i*80
            pygame.draw.rect(screen, (15,25,40), (W//2-280, y-10, 560, 60), border_radius=8)
            pygame.draw.rect(screen, (30,60,100), (W//2-280, y-10, 560, 60), width=1, border_radius=8)
            k_surf = font.render(key, True, col)
            n_surf = font.render(name, True, (220,230,240))
            d_surf = font_sm.render(desc, True, (100,130,160))
            screen.blit(k_surf, (W//2-260, y+4))
            screen.blit(n_surf, (W//2-200, y+4))
            screen.blit(d_surf, (W//2-200, y+28))

        hint = font_sm.render("Press 1 / 2 / 3 to launch", True, (60,100,130))
        screen.blit(hint, (W//2-hint.get_width()//2, 610))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_1: return 1, "urban"
                elif event.key == K_2: return 1, "forest"
                elif event.key == K_3: return 2, "urban"
    return 1, "urban"

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    pygame.init()
    pygame.display.set_caption("Drone Sim v2.0 — Realistic Environments")
    screen = pygame.display.set_mode((W, H))

    map_type, scene_variant = select_map_menu(screen)

    memory    = MemoryMap(GRID_CELLS)
    delivery_coords = None

    if map_type == 2:
        aes = AddressEntryScreen(screen)
        p_street, d_street = aes.run()
        if p_street is None:
            return
        delivery_coords = (STREET_REGISTRY[p_street], STREET_REGISTRY[d_street])
        obstacles, targets = generate_scene(memory, map_type, scene_variant, delivery_coords)
    elif map_type == 3:
        obstacles, targets = generate_scene(memory, map_type, scene_variant)
        ads = AreaDrawingScreen(screen, memory)
        rect = ads.run()
        if rect is None:
            return
        memory.bounding_box = rect
    else:
        obstacles, targets = generate_scene(memory, map_type, scene_variant)

    screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    clock  = pygame.time.Clock()

    glViewport(0, 0, W, H)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, W/H, 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    drone     = AIDrone(memory, map_type)
    drone.total_targets = len([t for t in targets if t.target_type != "dropoff"])
    camera    = Camera()
    hud       = HUD()
    hud_surf  = pygame.Surface((W, H), pygame.SRCALPHA)

    fireflies = FireflySystem(70) if scene_variant == "forest" else None

    paused     = False
    elapsed    = 0.0
    mouse_drag = False
    last_mouse = (0, 0)
    t          = 0.0

    # is_lit check
    def is_lit(x, z):
        if map_type in (2,3) and scene_variant=="urban": return True
        return math.hypot(x-drone.pos[0], z-drone.pos[2]) < SPOTLIGHT_R*1.1

    running = True
    while running:
        dt = min(clock.tick(60)/1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == QUIT: running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE: running = False
                elif event.key == K_SPACE: paused = not paused
                elif event.key == K_r:
                    memory = MemoryMap(GRID_CELLS)
                    obstacles, targets = generate_scene(memory, map_type, scene_variant)
                    drone  = AIDrone(memory, map_type)
                    drone.total_targets = len([t for t in targets if t.target_type!="dropoff"])
                    elapsed = 0.0; t = 0.0
                    fireflies = FireflySystem(70) if scene_variant=="forest" else None
                elif event.key == K_c:
                    camera.mode = (camera.mode+1)%3
                    print(f"  Camera: {['Follow','Top-Down','Free Orbit'][camera.mode]}")
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1: mouse_drag=True; last_mouse=event.pos
                elif event.button == 4: camera.dist=max(5, camera.dist-1.5)
                elif event.button == 5: camera.dist=min(60, camera.dist+1.5)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1: mouse_drag=False
            elif event.type == MOUSEMOTION:
                if mouse_drag:
                    dx=event.pos[0]-last_mouse[0]; dy=event.pos[1]-last_mouse[1]
                    camera.yaw  += dx*0.4
                    camera.pitch = np.clip(camera.pitch-dy*0.3, 5, 85)
                    last_mouse   = event.pos

        if not paused:
            t       += dt
            elapsed += dt
            drone.update(dt, obstacles, targets)
            for tgt in targets:
                tgt.update(dt)
                if tgt.found:
                    cx, cz = memory.world_to_cell(tgt.x, tgt.z)
                    memory.target_found[cx, cz] = 1
            if fireflies: fireflies.update(dt)

        # Clear
        if scene_variant == "forest":
            glClearColor(0.005, 0.010, 0.005, 1.0)
        elif map_type in (2,3):
            glClearColor(0.5, 0.7, 0.9, 1.0)
        else:
            glClearColor(*COLORS["bg"], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        camera.apply(drone.pos, dt)
        setup_lighting(drone.pos, map_type, scene_variant)

        draw_stars(map_type, scene_variant)
        draw_ground(WORLD_SIZE, map_type, scene_variant)
        draw_spotlight_pool(drone.pos[0], drone.pos[2], map_type, scene_variant)

        # Draw obstacles (buildings or trees)
        for obs in obstacles:
            lit = is_lit(obs.x, obs.z)
            if isinstance(obs, Tree):
                obs.draw(lit, t)
            elif isinstance(obs, FallenLog):
                obs.draw(lit)
            elif isinstance(obs, RealisticBuilding):
                obs.draw(lit)

        # Targets
        for tgt in targets:
            tgt.draw(is_lit(tgt.x, tgt.z), t)

        # Fireflies
        if fireflies:
            fireflies.draw()

        # Drone
        glDisable(GL_LIGHTING)
        drone.draw(t)
        glEnable(GL_LIGHTING)

        # HUD overlay
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        hud_surf.fill((0,0,0,0))
        hud.draw(hud_surf, drone, memory,
                 len([t2 for t2 in targets if t2.target_type!="dropoff"]),
                 elapsed, paused, scene_variant)

        hud_data = pygame.image.tostring(hud_surf, "RGBA", False)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1,1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0)
        glTexCoord2f(1,0); glVertex2f(W,0)
        glTexCoord2f(1,1); glVertex2f(W,H)
        glTexCoord2f(0,1); glVertex2f(0,H)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex_id])

        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

        pygame.display.flip()

    pygame.quit()
    print(f"\n  Mission ended.")
    print(f"  Survivors rescued: {drone.targets_found}/{drone.total_targets}")
    print(f"  Distance flown:    {drone.dist_traveled:.1f} m")
    print(f"  Map coverage:      {memory.coverage():.1f}%")

if __name__ == "__main__":
    main()