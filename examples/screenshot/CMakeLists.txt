buildExample(screenshot screenshot.cpp)

compile_shaders(screenshot
    shaders/mesh.frag
    shaders/mesh.vert
)

link_assets(screenshot
    models/chinesedragon.dae
)

