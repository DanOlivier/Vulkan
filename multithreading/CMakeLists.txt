buildExample(multithreading multithreading.cpp)

compile_shaders(multithreading
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/phong.frag
    shaders/phong.vert
    shaders/starsphere.frag
    shaders/starsphere.vert
)

copy_assets(multithreading
    models/retroufo_red_lowpoly.dae
    models/sphere.obj
)
