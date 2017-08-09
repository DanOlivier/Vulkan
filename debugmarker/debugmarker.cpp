/*
* Vulkan Example - Example for VK_EXT_debug_marker extension. To be used in conjuction with a debugging app like RenderDoc (https://renderdoc.org)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Offscreen properties
#define OFFSCREEN_DIM 256
#define OFFSCREEN_FORMAT vk::Format::eR8G8B8A8Unorm
#define OFFSCREEN_FILTER vk::Filter::eLinear;

// Setup and functions for the VK_EXT_debug_marker_extension
// Extension spec can be found at https://github.com/KhronosGroup/Vulkan-Docs/blob/1.0-VK_EXT_debug_marker/doc/specs/vulkan/appendices/VK_EXT_debug_marker.txt
// Note that the extension will only be present if run from an offline debugging application
namespace DebugMarker
{
	bool active = false;
	bool extensionPresent = false;

	PFN_vkDebugMarkerSetObjectTagEXT vkDebugMarkerSetObjectTag = nullptr;
	PFN_vkDebugMarkerSetObjectNameEXT vkDebugMarkerSetObjectName = nullptr;
	PFN_vkCmdDebugMarkerBeginEXT vkCmdDebugMarkerBegin = nullptr;
	PFN_vkCmdDebugMarkerEndEXT vkCmdDebugMarkerEnd = nullptr;
	PFN_vkCmdDebugMarkerInsertEXT vkCmdDebugMarkerInsert = nullptr;

	// Get function pointers for the debug report extensions from the device
	void setup(vk::Device device, vk::PhysicalDevice physicalDevice)
	{
		// Check if the debug marker extension is present (which is the case if run from a graphics debugger)
		std::vector<vk::ExtensionProperties> extensions = physicalDevice.enumerateDeviceExtensionProperties();
		for (auto extension : extensions) {
			if (strcmp(extension.extensionName, VK_EXT_DEBUG_MARKER_EXTENSION_NAME) == 0) {
				extensionPresent = true;
				break;
			}
		}

		if (extensionPresent) {
			// The debug marker extension is not part of the core, so function pointers need to be loaded manually
			vkDebugMarkerSetObjectTag = (PFN_vkDebugMarkerSetObjectTagEXT)vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectTagEXT");
			vkDebugMarkerSetObjectName = (PFN_vkDebugMarkerSetObjectNameEXT)vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT");
			vkCmdDebugMarkerBegin = (PFN_vkCmdDebugMarkerBeginEXT)vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT");
			vkCmdDebugMarkerEnd = (PFN_vkCmdDebugMarkerEndEXT)vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT");
			vkCmdDebugMarkerInsert = (PFN_vkCmdDebugMarkerInsertEXT)vkGetDeviceProcAddr(device, "vkCmdDebugMarkerInsertEXT");
			// Set flag if at least one function pointer is present
			active = (!!vkDebugMarkerSetObjectName);
		}
		else {
			std::cout << "Warning: " << VK_EXT_DEBUG_MARKER_EXTENSION_NAME << " not present, debug markers are disabled.";
			std::cout << "Try running from inside a Vulkan graphics debugger (e.g. RenderDoc)" << std::endl;
		}
	}

	// Sets the debug name of an object
	// All Objects in Vulkan are represented by their 64-bit handles which are passed into this function
	// along with the object type
	void setObjectName(vk::Device device, uint64_t object, VkDebugReportObjectTypeEXT objectType, const char *name)
	{
		// Check for valid function pointer (may not be present if not running in a debugging application)
		if (active)
		{
			VkDebugMarkerObjectNameInfoEXT nameInfo = {};
			nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
			nameInfo.objectType = objectType;
			nameInfo.object = object;
			nameInfo.pObjectName = name;
			vkDebugMarkerSetObjectName(device, &nameInfo);
		}
	}

	// Set the tag for an object
	void setObjectTag(vk::Device device, uint64_t object, VkDebugReportObjectTypeEXT objectType, uint64_t name, size_t tagSize, const void* tag)
	{
		// Check for valid function pointer (may not be present if not running in a debugging application)
		if (active)
		{
			VkDebugMarkerObjectTagInfoEXT tagInfo = {};
			tagInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT;
			tagInfo.objectType = objectType;
			tagInfo.object = object;
			tagInfo.tagName = name;
			tagInfo.tagSize = tagSize;
			tagInfo.pTag = tag;
			vkDebugMarkerSetObjectTag(device, &tagInfo);
		}
	}

	// Start a new debug marker region
	void beginRegion(vk::CommandBuffer cmdbuffer, const char* pMarkerName, glm::vec4 color)
	{
		// Check for valid function pointer (may not be present if not running in a debugging application)
		if (active)
		{
			VkDebugMarkerMarkerInfoEXT markerInfo = {};
			markerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
			memcpy(markerInfo.color, &color[0], sizeof(float) * 4);
			markerInfo.pMarkerName = pMarkerName;
			vkCmdDebugMarkerBegin(cmdbuffer, &markerInfo);
		}
	}

	// Insert a new debug marker into the command buffer
	void insert(vk::CommandBuffer cmdbuffer, std::string markerName, glm::vec4 color)
	{
		// Check for valid function pointer (may not be present if not running in a debugging application)
		if (active)
		{
			VkDebugMarkerMarkerInfoEXT markerInfo = {};
			markerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
			memcpy(markerInfo.color, &color[0], sizeof(float) * 4);
			markerInfo.pMarkerName = markerName.c_str();
			vkCmdDebugMarkerInsert(cmdbuffer, &markerInfo);
		}
	}

	// End the current debug marker region
	void endRegion(vk::CommandBuffer cmdBuffer)
	{
		// Check for valid function (may not be present if not runnin in a debugging application)
		if (vkCmdDebugMarkerEnd)
		{
			vkCmdDebugMarkerEnd(cmdBuffer);
		}
	}
};

// Vertex layout for the models
vks::VertexLayout vertexLayout = vks::VertexLayout({
	vks::VERTEX_COMPONENT_POSITION,
	vks::VERTEX_COMPONENT_NORMAL,
	vks::VERTEX_COMPONENT_UV,
	vks::VERTEX_COMPONENT_COLOR,
});

struct Scene {

	vks::Model model;
	std::vector<std::string> modelPartNames;

	void draw(vk::CommandBuffer cmdBuffer)
	{
		std::vector<vk::DeviceSize> offsets = { 0 };
		cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, model.vertices.buffer, offsets);
		cmdBuffer.bindIndexBuffer(model.indices.buffer, 0, vk::IndexType::eUint32);
		for (uint32_t i = 0; i < model.parts.size(); i++)
		{
			// Add debug marker for mesh name
			DebugMarker::insert(cmdBuffer, "Draw \"" + modelPartNames[i] + "\"", glm::vec4(0.0f));
			cmdBuffer.drawIndexed(model.parts[i].indexCount, 1, model.parts[i].indexBase, 0, 0);
		}
	}

	void loadFromFile(std::string filename, vks::VulkanDevice* vulkanDevice, vk::Queue queue)
	{
		model.loadFromFile(filename, vertexLayout, 1.0f, vulkanDevice, queue);
	}
};

class VulkanExample : public VulkanExampleBase
{
public:
	bool wireframe = true;
	bool glow = true;

	Scene scene, sceneGlow;

	vks::Buffer uniformBuffer;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(0.0f, 5.0f, 15.0f, 1.0f);
	} uboVS;

	struct Pipelines {
		vk::Pipeline toonshading;
		vk::Pipeline color;
		vk::Pipeline wireframe;
		vk::Pipeline postprocess;
	} pipelines;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSetLayout descriptorSetLayout;

	struct {
		vk::DescriptorSet scene;
		vk::DescriptorSet fullscreen;
	} descriptorSets;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		vk::Image image;
		vk::DeviceMemory mem;
		vk::ImageView view;
	};
	struct OffscreenPass {
		uint32_t width, height;
		vk::Framebuffer frameBuffer;
		FrameBufferAttachment color, depth;
		vk::RenderPass renderPass;
		vk::Sampler sampler;
		vk::DescriptorImageInfo descriptor;
		vk::CommandBuffer commandBuffer;
		// Semaphore used to synchronize between offscreen and final scene render pass
		vk::Semaphore semaphore;
	} offscreenPass;

	// Random tag data
	struct DemoTag {
		const char name[17] = "debug marker tag";
	} demoTag;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -8.5f;
		zoomSpeed = 2.5f;
		rotationSpeed = 0.5f;
		rotation = { -4.35f, 16.25f, 0.0f };
		cameraPos = { 0.1f, 1.1f, 0.0f };
		enableTextOverlay = true;
		title = "Vulkan Example - VK_EXT_debug_marker";
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
		wireframe = deviceFeatures.fillModeNonSolid;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class
		device.destroyPipeline(pipelines.toonshading);
		device.destroyPipeline(pipelines.color);
		device.destroyPipeline(pipelines.postprocess);
		if (pipelines.wireframe) {
			device.destroyPipeline(pipelines.wireframe);
		}

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		// Destroy and free mesh resources 
		scene.model.destroy();
		sceneGlow.model.destroy();

		uniformBuffer.destroy();

		// Offscreen
		// Color attachment
		device.destroyImageView(offscreenPass.color.view);
		device.destroyImage(offscreenPass.color.image);
		device.freeMemory(offscreenPass.color.mem);

		// Depth attachment
		device.destroyImageView(offscreenPass.depth.view);
		device.destroyImage(offscreenPass.depth.image);
		device.freeMemory(offscreenPass.depth.mem);

		device.destroyRenderPass(offscreenPass.renderPass);
		device.destroySampler(offscreenPass.sampler);
		device.destroyFramebuffer(offscreenPass.frameBuffer);

		device.freeCommandBuffers(cmdPool, offscreenPass.commandBuffer);
		device.destroySemaphore(offscreenPass.semaphore);
	}

	// Prepare a texture target and framebuffer for offscreen rendering
	void prepareOffscreen()
	{
		offscreenPass.width = OFFSCREEN_DIM;
		offscreenPass.height = OFFSCREEN_DIM;

		// Find a suitable depth format
		vk::Format fbDepthFormat;
		vk::Bool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
		assert(validDepthFormat);

		// Color attachment
		vk::ImageCreateInfo image;
		image.imageType = vk::ImageType::e2D;
		image.format = OFFSCREEN_FORMAT;
		image.extent = vk::Extent3D{ offscreenPass.width, offscreenPass.height, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = vk::SampleCountFlagBits::e1;
		image.tiling = vk::ImageTiling::eOptimal;
		// We will sample directly from the color attachment
		image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

		vk::MemoryAllocateInfo memAlloc;
		vk::MemoryRequirements memReqs;

		offscreenPass.color.image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(offscreenPass.color.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		offscreenPass.color.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(offscreenPass.color.image, offscreenPass.color.mem, 0);

		vk::ImageViewCreateInfo colorImageView;
		colorImageView.viewType = vk::ImageViewType::e2D;
		colorImageView.format = OFFSCREEN_FORMAT;
		colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;
		colorImageView.image = offscreenPass.color.image;
		offscreenPass.color.view = device.createImageView(colorImageView);

		// Create sampler to sample from the attachment in the fragment shader
		vk::SamplerCreateInfo samplerInfo;
		samplerInfo.magFilter = OFFSCREEN_FILTER;
		samplerInfo.minFilter = OFFSCREEN_FILTER;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		offscreenPass.sampler = device.createSampler(samplerInfo);

		// Depth stencil attachment
		image.format = fbDepthFormat;
		image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

		offscreenPass.depth.image = device.createImage(image);
		memReqs = device.getImageMemoryRequirements(offscreenPass.depth.image);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		offscreenPass.depth.mem = device.allocateMemory(memAlloc);
		device.bindImageMemory(offscreenPass.depth.image, offscreenPass.depth.mem, 0);

		vk::ImageViewCreateInfo depthStencilView;
		depthStencilView.viewType = vk::ImageViewType::e2D;
		depthStencilView.format = fbDepthFormat;
		//depthStencilView.flags = 0;
		//depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = offscreenPass.depth.image;
		offscreenPass.depth.view = device.createImageView(depthStencilView);

		// Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

		std::array<vk::AttachmentDescription, 2> attchmentDescriptions = {};
		// Color attachment
		attchmentDescriptions[0].format = OFFSCREEN_FORMAT;
		attchmentDescriptions[0].samples = vk::SampleCountFlagBits::e1;
		attchmentDescriptions[0].loadOp = vk::AttachmentLoadOp::eClear;
		attchmentDescriptions[0].storeOp = vk::AttachmentStoreOp::eStore;
		attchmentDescriptions[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attchmentDescriptions[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attchmentDescriptions[0].initialLayout = vk::ImageLayout::eUndefined;
		attchmentDescriptions[0].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		// Depth attachment
		attchmentDescriptions[1].format = fbDepthFormat;
		attchmentDescriptions[1].samples = vk::SampleCountFlagBits::e1;
		attchmentDescriptions[1].loadOp = vk::AttachmentLoadOp::eClear;
		attchmentDescriptions[1].storeOp = vk::AttachmentStoreOp::eDontCare;
		attchmentDescriptions[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		attchmentDescriptions[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		attchmentDescriptions[1].initialLayout = vk::ImageLayout::eUndefined;
		attchmentDescriptions[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };
		vk::AttachmentReference depthReference = { 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };

		vk::SubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<vk::SubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
		dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
		dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

		// Create the actual renderpass
		vk::RenderPassCreateInfo renderPassInfo = {};

		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		offscreenPass.renderPass = device.createRenderPass(renderPassInfo);

		vk::ImageView attachments[2];
		attachments[0] = offscreenPass.color.view;
		attachments[1] = offscreenPass.depth.view;

		vk::FramebufferCreateInfo fbufCreateInfo;
		fbufCreateInfo.renderPass = offscreenPass.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = offscreenPass.width;
		fbufCreateInfo.height = offscreenPass.height;
		fbufCreateInfo.layers = 1;

		offscreenPass.frameBuffer = device.createFramebuffer(fbufCreateInfo);

		// Fill a descriptor for later use in a descriptor set 
		offscreenPass.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		offscreenPass.descriptor.imageView = offscreenPass.color.view;
		offscreenPass.descriptor.sampler = offscreenPass.sampler;

		// Name some objects for debugging
		DebugMarker::setObjectName(device, (uint64_t)VkImage(offscreenPass.color.image), VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, "Off-screen color framebuffer");
		DebugMarker::setObjectName(device, (uint64_t)VkImage(offscreenPass.depth.image), VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, "Off-screen depth framebuffer");
		DebugMarker::setObjectName(device, (uint64_t)VkSampler(offscreenPass.sampler), VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT, "Off-screen framebuffer default sampler");
	}

	// Command buffer for rendering color only scene for glow
	void buildOffscreenCommandBuffer()
	{
		if (!offscreenPass.commandBuffer)
		{
			offscreenPass.commandBuffer = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		}
		if (!offscreenPass.semaphore)
		{
			// Create a semaphore used to synchronize offscreen rendering and usage
			vk::SemaphoreCreateInfo semaphoreCreateInfo;
			offscreenPass.semaphore = device.createSemaphore(semaphoreCreateInfo);
		}

		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = offscreenPass.renderPass;
		renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
		renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		offscreenPass.commandBuffer.begin(cmdBufInfo);

		// Start a new debug marker region
		DebugMarker::beginRegion(offscreenPass.commandBuffer, "Off-screen scene rendering", glm::vec4(1.0f, 0.78f, 0.05f, 1.0f));

		vk::Viewport viewport = vks::initializers::viewport((float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
		offscreenPass.commandBuffer.setViewport(0, viewport);

		vk::Rect2D scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
		offscreenPass.commandBuffer.setScissor(0, scissor);

		offscreenPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.scene, nullptr);
		offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.color);

		// Draw glow scene
		sceneGlow.draw(offscreenPass.commandBuffer);

		offscreenPass.commandBuffer.endRenderPass();

		DebugMarker::endRegion(offscreenPass.commandBuffer);

		offscreenPass.commandBuffer.end();
	}

	void loadScene()
	{
		scene.loadFromFile(getAssetPath() + "models/treasure_smooth.dae", vulkanDevice, queue);
		sceneGlow.loadFromFile(getAssetPath() + "models/treasure_glow.dae", vulkanDevice, queue);

		// Name the meshes
		// ASSIMP does not load mesh names from the COLLADA file used in this example so we need to set them manually
		// These names are used in command buffer creation for setting debug markers
		std::vector<std::string> names = { "hill", "crystals", "rocks", "cave", "tree", "mushroom stems", "blue mushroom caps", "red mushroom caps", "grass blades", "chest box", "chest fittings" };
		for (size_t i = 0; i < names.size(); i++) {
			scene.modelPartNames.push_back(names[i]);
			sceneGlow.modelPartNames.push_back(names[i]);
		}

		// Name the buffers for debugging
		// Scene
		DebugMarker::setObjectName(device, (uint64_t)VkBuffer(scene.model.vertices.buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Scene vertex buffer");
		DebugMarker::setObjectName(device, (uint64_t)VkBuffer(scene.model.indices.buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Scene index buffer");
		// Glow
		DebugMarker::setObjectName(device, (uint64_t)VkBuffer(sceneGlow.model.vertices.buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Glow vertex buffer");
		DebugMarker::setObjectName(device, (uint64_t)VkBuffer(sceneGlow.model.indices.buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Glow index buffer");
	}

	void reBuildCommandBuffers()
	{
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}
		buildCommandBuffers();
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			// Start a new debug marker region
			DebugMarker::beginRegion(drawCmdBuffers[i], "Render scene", glm::vec4(0.5f, 0.76f, 0.34f, 1.0f));

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(wireframe ? width / 2 : width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.scene, nullptr);

			// Solid rendering

			// Start a new debug marker region
			DebugMarker::beginRegion(drawCmdBuffers[i], "Toon shading draw", glm::vec4(0.78f, 0.74f, 0.9f, 1.0f));

			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.toonshading);
			scene.draw(drawCmdBuffers[i]);

			DebugMarker::endRegion(drawCmdBuffers[i]);

			// Wireframe rendering
			if (wireframe)
			{
				// Insert debug marker
				DebugMarker::beginRegion(drawCmdBuffers[i], "Wireframe draw", glm::vec4(0.53f, 0.78f, 0.91f, 1.0f));

				scissor.offset.x = width / 2;
				drawCmdBuffers[i].setScissor(0, scissor);

				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.wireframe);
				scene.draw(drawCmdBuffers[i]);

				DebugMarker::endRegion(drawCmdBuffers[i]);

				scissor.offset.x = 0;
				scissor.extent.width = width;
				drawCmdBuffers[i].setScissor(0, scissor);
			}

			// Post processing
			if (glow)
			{
				DebugMarker::beginRegion(drawCmdBuffers[i], "Apply post processing", glm::vec4(0.93f, 0.89f, 0.69f, 1.0f));

				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.postprocess);
				// Full screen quad is generated by the vertex shaders, so we reuse four vertices (for four invocations) from current vertex buffer
				vkCmdDraw(drawCmdBuffers[i], 4, 1, 0, 0);

				DebugMarker::endRegion(drawCmdBuffers[i]);
			}


			drawCmdBuffers[i].endRenderPass();

			// End current debug marker region
			DebugMarker::endRegion(drawCmdBuffers[i]);

			drawCmdBuffers[i].end();
		}
	}


	void setupDescriptorPool()
	{
		// Example uses one ubo and one combined image sampler
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				1);

		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eVertex,
				0),
			// Binding 1 : Fragment shader combined sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler,
				vk::ShaderStageFlagBits::eFragment,
				1),
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Name for debugging
		DebugMarker::setObjectName(device, (uint64_t)VkPipelineLayout(pipelineLayout), VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT, "Shared pipeline layout");
		DebugMarker::setObjectName(device, (uint64_t)VkDescriptorSetLayout(descriptorSetLayout), VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT, "Shared descriptor set layout");
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		descriptorSets.scene = device.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.scene,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffer.descriptor),
			// Binding 1 : Color map 
			vks::initializers::writeDescriptorSet(
				descriptorSets.scene,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&offscreenPass.descriptor)
		};

		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				vk::PipelineInputAssemblyStateCreateFlags(),
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState();

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_TRUE,
				VK_TRUE,
				vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		// Shared vertex inputs

		// Binding description
		vk::VertexInputBindingDescription vertexInputBinding =
			vks::initializers::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vertexLayout.stride(), vk::VertexInputRate::eVertex);

		// Attribute descriptions
		// Describes memory layout and shader positions
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, vk::Format::eR32G32B32Sfloat, 0),						// Location 0: Position		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),		// Location 1: Normal		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, vk::Format::eR32G32Sfloat, sizeof(float) * 6),			// Location 2: Texture coordinates		
			vks::initializers::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3, vk::Format::eR32G32B32Sfloat, sizeof(float) * 8),		// Location 3: Color		
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Toon shading pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/toon.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/toon.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.toonshading = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Color only pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelineCreateInfo.renderPass = offscreenPass.renderPass;
		pipelines.color = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid)
		{
			rasterizationState.polygonMode = vk::PolygonMode::eLine;
			pipelineCreateInfo.renderPass = renderPass;
			pipelines.wireframe = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		}

		// Post processing effect
		shaderStages[0] = loadShader(getAssetPath() + "shaders/postprocess.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/postprocess.frag.spv", vk::ShaderStageFlagBits::eFragment);
		depthStencilState.depthTestEnable = VK_FALSE;
		depthStencilState.depthWriteEnable = VK_FALSE;
		rasterizationState.polygonMode = vk::PolygonMode::eFill;
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		blendAttachmentState.blendEnable =  VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;
		pipelines.postprocess = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Name shader moduels for debugging
		// Shader module count starts at 2 when text overlay in base class is enabled
		uint32_t moduleIndex = enableTextOverlay ? 2 : 0;
		DebugMarker::setObjectName(device, (uint64_t)VkShaderModule(shaderModules[moduleIndex + 0]), VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Toon shading vertex shader");
		DebugMarker::setObjectName(device, (uint64_t)VkShaderModule(shaderModules[moduleIndex + 1]), VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Toon shading fragment shader");
		DebugMarker::setObjectName(device, (uint64_t)VkShaderModule(shaderModules[moduleIndex + 2]), VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Color-only vertex shader");
		DebugMarker::setObjectName(device, (uint64_t)VkShaderModule(shaderModules[moduleIndex + 3]), VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Color-only fragment shader");
		DebugMarker::setObjectName(device, (uint64_t)VkShaderModule(shaderModules[moduleIndex + 4]), VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Postprocess vertex shader");
		DebugMarker::setObjectName(device, (uint64_t)VkShaderModule(shaderModules[moduleIndex + 5]), VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Postprocess fragment shader");

		// Name pipelines for debugging
		DebugMarker::setObjectName(device, (uint64_t)VkPipeline(pipelines.toonshading), VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT, "Toon shading pipeline");
		DebugMarker::setObjectName(device, (uint64_t)VkPipeline(pipelines.color), VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT, "Color only pipeline");
		DebugMarker::setObjectName(device, (uint64_t)VkPipeline(pipelines.wireframe), VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT, "Wireframe rendering pipeline");
		DebugMarker::setObjectName(device, (uint64_t)VkPipeline(pipelines.postprocess), VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT, "Post processing pipeline");
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffer,
			sizeof(uboVS));

		// Map persistent
		uniformBuffer.map();


		// Name uniform buffer for debugging
		DebugMarker::setObjectName(device, (uint64_t)VkBuffer(uniformBuffer.buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Scene uniform buffer block");
		// Add some random tag
		DebugMarker::setObjectTag(device, (uint64_t)VkBuffer(uniformBuffer.buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, 0, sizeof(demoTag), &demoTag);

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		uboVS.projection = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f);
		glm::mat4 viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));

		uboVS.model = viewMatrix * glm::translate(glm::mat4(), cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffer.mapped, &uboVS, sizeof(uboVS));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Offscreen rendering
		if (glow)
		{
			// Wait for swap chain presentation to finish
			submitInfo.pWaitSemaphores = &semaphores.presentComplete;
			// Signal ready with offscreen semaphore
			submitInfo.pSignalSemaphores = &offscreenPass.semaphore;

			// Submit work
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &offscreenPass.commandBuffer;
			queue.submit(submitInfo, vk::Fence(nullptr));
		}

		// Scene rendering
		// Wait for offscreen semaphore
		submitInfo.pWaitSemaphores = &offscreenPass.semaphore;
		// Signal ready with render complete semaphpre
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;

		// Submit work
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		DebugMarker::setup(device, physicalDevice);
		loadScene();
		prepareOffscreen();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		buildOffscreenCommandBuffer();
		updateTextOverlay();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case 0x57:
		case GAMEPAD_BUTTON_X:
			if (deviceFeatures.fillModeNonSolid) 
			{
				wireframe = !wireframe;
				reBuildCommandBuffers();
			}
			break;
		case 0x47:
		case GAMEPAD_BUTTON_A:
			glow = !glow;
			reBuildCommandBuffers();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		if (DebugMarker::active)
		{
			textOverlay->addText("VK_EXT_debug_marker active", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		}
		else
		{
			textOverlay->addText("VK_EXT_debug_marker not present", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		}
	}
};

VULKAN_EXAMPLE_MAIN()