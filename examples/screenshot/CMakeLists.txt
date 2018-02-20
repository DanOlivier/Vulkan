buildExample(screenshot screenshot.cpp)

compile_shaders(screenshot
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/mesh.frag
    shaders/mesh.vert
)

copy_assets(screenshot
    models/chinesedragon.dae
)

