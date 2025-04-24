import numpy as np
import threading
from realsense import RealsenseCamera

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import OpenGL.arrays.vbo as glvbo

def normalize_depth(z, z_min, z_max):
    """Normalize depth values to the range [0, 1]."""
    return np.clip((z - z_min) / (z_max - z_min), 0, 1)

def depth_to_color(depth_norm):
    """Map normalized depth to RGB using a simple gradient."""
    # Example: blue to red gradient
    r = depth_norm
    g = 1.0 - depth_norm
    b = 0.5 * (1.0 - depth_norm)
    return r, g, b

def update_vertices():
    global vertices, camera, update_lock, updating
    while updating:
        cf, df, ci, di, dc, vi = camera.get_next_frame()
        vertices = vi.reshape(-1,3)

def depth2vertex(depth_frame):
    points = pc.calculate(depth_frame)
    depth_img = np.asanyarray(depth_frame.get_data())
    v,t = points.get_vertices(), points.get_texture_coordinates()
    vertices = np.asanyarray(v).view(np.float32).reshape(-1,3)
    vertex_image = vertices.reshape((*depth_img.shape,3))
    return vertices, vertex_image

def initialize_vbo(vertices):
    global vbo, vertex_count
    print('init vbo')
    vbo = glvbo.VBO(vertices.flatten())
    vertex_count = 0
    print('done init vbo')

def update_vbo(vertices):
    global vbo, vertex_count
    print('update vbo')
    print(vertices.shape)
    if vertices is not None:
        vertex_count = len(vertices)
        vbo.set_array(vertices.flatten())
        vbo.bind()
    print('done update vbo')

def display():
    global vbo, vertex_count, vertices
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()
    glu.gluLookAt(0, 0, 0, 0, 0, 5, 0, 1, 0)

    update_vbo(vertices)

    if vertex_count > 0:
        vbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, vbo)

        # Apply colors based on depth
        colors = np.array(
            [depth_to_color(normalize_depth(z, *depth_range)) for x, y, z in vertices], dtype=np.float32
        )
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, colors)

        gl.glDrawArrays(gl.GL_POINTS, 0, vertex_count)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        vbo.unbind()

    glut.glutSwapBuffers()

def idle():
    glut.glutPostRedisplay()

def main():
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"Real-Time 3D Point Cloud")
    gl.glEnable(gl.GL_DEPTH_TEST)

    while vertices is None: continue
    initialize_vbo(vertices)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60, 800 / 600, 0.1, 100.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)

    glut.glutDisplayFunc(display)
    glut.glutIdleFunc(idle)
    glut.glutMainLoop()

if __name__ == "__main__":
    vbo = None
    vertices = None
    depth_range = (0.1, 10.0)

    update_lock = threading.Lock()
    updating = True

    camera = RealsenseCamera()
    camera.setAveragingCount(10)

    update_thread = threading.Thread(target=update_vertices)
    update_thread.start()

    try:
        main()
    finally:
        updating = False
        update_thread.join()
        camera.stop()
