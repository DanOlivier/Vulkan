vkCreateImage + vkGetImageMemoryRequirements + vkAllocateMemory:
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in examples/deferred/deferred.cpp:275
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in examples/deferredmultisampling/deferredmultisampling.cpp:282

deferredshadow:
vkCreateImage + vkGetImageMemoryRequirements + vkAllocateMemory
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in base/VulkanFrameBuffer.hpp:185
hdr:
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in ../base/VulkanTexture.hpp at line 898

All use VK_FORMAT_R16G16B16A16_SFLOAT

vkCreateImage + vkGetImageMemoryRequirements + 
  vulkanDevice->getMemoryType + vks::initializers::memoryAllocateInfo +
  vkAllocateMemory
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in examples/multisampling/multisampling.cpp:226 (in setupMultisampleTarget, using a depthFormat of VK_FORMAT_D32_SFLOAT_S8_UINT)
Note: VulkanExampleBase::depthFormat is initialized in VulkanExampleBase::initVulkan.
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in examples/shadowmappingcascade/shadowmappingcascade.cpp:351 (in prepareDepthPass, using VK_FORMAT_D32_SFLOAT_S8_UINT)
Note: redundant call to vks::tools::getSupportedDepthFormat to initialize local depthFormat
Fatal : VkResult is "ERROR_OUT_OF_DEVICE_MEMORY" in examples/pbrtexture/main.cpp:986 (in generatePrefilteredCube, as we create an image with VK_FORMAT_R16G16B16A16_SFLOAT)

pipelinestatistics: blank?!

Resizing, i.e.:
Fatal : VkResult is "ERROR_OUT_OF_DATE_KHR" in examples/triangle/triangle.cpp:358
queuePresent

vscode: "Unable to start debugging. An item with the same key has already been added"
Caused by VULKAN_EXAMPLE_MAIN macro, which puts multiple functions on the same line as main (handleEvent, which is flagged as unused by the compiler)