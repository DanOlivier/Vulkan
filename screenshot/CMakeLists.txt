buildExample(screenshot screenshot.cpp)

compile_shaders(screenshot
    shaders/mesh.frag
    shaders/mesh.vert
)

copy_assets(screenshot
    models/chinesedragon.dae
)

