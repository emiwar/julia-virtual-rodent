#import GLAbstraction
#import GLFW
import MuJoCo
MuJoCo.init_visualiser()

function GLFWRender(model, data)
    GLFW.WindowHint(GLFW.SAMPLES, 4)
    window = GLFW.Window(name="Test", resolution=(600,600))
    #GLAbstraction.set_context!(window)
    GLFW.MakeContextCurrent(window)
    GLFW.SwapInterval(1)

    scn = MuJoCo.VisualiserScene()
    cam = MuJoCo.VisualiserCamera()
    opt = MuJoCo.VisualiserOption()
    con = MuJoCo.RendererContext()
    pert = MuJoCo.VisualiserPerturb()
    #MuJoCo.mjv_defaultCamera(cam)
    #MuJoCo.mjv_defaultOption(opt)
    #MuJoCo.mjr_defaultContext(con)
    MuJoCo.mjv_makeScene(model, scn, 1000)
    MuJoCo.mjr_makeContext(model, con, MuJoCo.mjFONTSCALE_100)

    while !GLFW.WindowShouldClose(window)
        GLFW.glClear(GLAbstraction.GL_COLOR_BUFFER_BIT)

        w, h = GLFW.GetFramebufferSize(window)
        rect = MuJoCo.mjrRect(Cint(0), Cint(0), Cint(w), Cint(h))
        println(rect)
        #MuJoCo.mjv_updateScene(model, data, opt, pert, cam, MuJoCo.mjCAT_ALL, scn)
        #MuJoCo.mjr_render(rect, scn, con)

        GLFW.SwapBuffers(window)
        GLFW.PollEvents()
        if GLFW.GetKey(window, GLFW.KEY_ESCAPE) == GLFW.PRESS
            GLFW.SetWindowShouldClose(window, true)
        end
    end
    GLFW.DestroyWindow(window)
end

modelPath = "/home/emil/Development/custom_torchrl_env/models/rodent_with_floor.xml"
model = MuJoCo.load_model(modelPath)
data = MuJoCo.init_data(model)

GLFWRender(model, data)