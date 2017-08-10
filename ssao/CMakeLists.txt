buildExample(ssao ssao.cpp)

compile_shaders(ssao
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/blur.frag
    shaders/composition.frag
    shaders/fullscreen.vert
    shaders/gbuffer.frag
    shaders/gbuffer.vert
    shaders/ssao.frag
)

copy_assets(ssao
    models/sibenik/sibenik.dae
)

