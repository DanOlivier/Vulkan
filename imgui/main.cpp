/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanExampleBase.hpp"
#include "VulkanDevice.hpp"
#include "VulkanModel.hpp"

#include <imgui.h>

#define ENABLE_VALIDATION false

// Options and values to display/toggle from the UI
struct UISettings {
	bool displayModels = true;
	bool displayLogos = true;
	bool displayBackground = true;
	bool animateLight = false;
	float lightSpeed = 0.25f;
	std::array<float, 50> frameTimes{};
	float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
	float lightTimer = 0.0f;
} uiSettings;

// ----------------------------------------------------------------------------
// ImGUI class
// ----------------------------------------------------------------------------
class ImGUI {
private:
	// Vulkan resources for rendering the UI
	vk::Sampler sampler;
	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	int32_t vertexCount = 0;
	int32_t indexCount = 0;
	vk::DeviceMemory fontMemory;
	vk::Image fontImage;
	vk::ImageView fontView;
	vk::PipelineCache pipelineCache;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline pipeline;
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;
	vks::VulkanDevice *device;
	VulkanExampleBase *example;
public:
	// UI params are set via push constants
	struct PushConstBlock {
		glm::vec2 scale;
		glm::vec2 translate;
	} pushConstBlock;

	ImGUI(VulkanExampleBase *example) : example(example) 
	{
		device = example->vulkanDevice;
	};
	
	~ImGUI()
	{
		// Release all Vulkan resources required for rendering imGui
		vertexBuffer.destroy();
		indexBuffer.destroy();
		device->logicalDevice.destroyImage(fontImage);
		device->logicalDevice.destroyImageView(fontView);
		device->logicalDevice.freeMemory(fontMemory);
		device->logicalDevice.destroySampler(sampler);
		device->logicalDevice.destroyPipelineCache(pipelineCache);
		device->logicalDevice.destroyPipeline(pipeline);
		device->logicalDevice.destroyPipelineLayout(pipelineLayout);
		device->logicalDevice.destroyDescriptorPool(descriptorPool);
		device->logicalDevice.destroyDescriptorSetLayout(descriptorSetLayout);
	}

	// Initialize styles, keys, etc.
	void init(float width, float height)
	{
		// Color scheme
		ImGuiStyle& style = ImGui::GetStyle();
		style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
		style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
		style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
		style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
		style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
		// Dimensions
		ImGuiIO& io = ImGui::GetIO();
		io.DisplaySize = ImVec2(width, height);
		io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
	}

	// Initialize all Vulkan resources used by the ui
	void initResources(vk::RenderPass renderPass, vk::Queue copyQueue)
	{
		ImGuiIO& io = ImGui::GetIO();

		// Create font texture
		unsigned char* fontData;
		int32_t texWidth, texHeight;
		io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);
		vk::DeviceSize uploadSize = texWidth*texHeight * 4 * sizeof(char);

		// Create target image for copy
		vk::ImageCreateInfo imageInfo;
		imageInfo.imageType = vk::ImageType::e2D;
		imageInfo.format = vk::Format::eR8G8B8A8Unorm;
		imageInfo.extent = vk::Extent3D{ texWidth, texHeight, 1 };
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = vk::SampleCountFlagBits::e1;
		imageInfo.tiling = vk::ImageTiling::eOptimal;
		imageInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		imageInfo.sharingMode = vk::SharingMode::eExclusive;
		imageInfo.initialLayout = vk::ImageLayout::eUndefined;
		fontImage = device->logicalDevice.createImage(imageInfo);
		vk::MemoryRequirements memReqs;
		memReqs = device->logicalDevice.getImageMemoryRequirements(fontImage);
		vk::MemoryAllocateInfo memAllocInfo;
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		fontMemory = device->logicalDevice.allocateMemory(memAllocInfo);
		device->logicalDevice.bindImageMemory(fontImage, fontMemory, 0);

		// Image view
		vk::ImageViewCreateInfo viewInfo;
		viewInfo.image = fontImage;
		viewInfo.viewType = vk::ImageViewType::e2D;
		viewInfo.format = vk::Format::eR8G8B8A8Unorm;
		viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;
		fontView = device->logicalDevice.createImageView(viewInfo);

		// Staging buffers for font data upload
		vks::Buffer stagingBuffer;

		device->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&stagingBuffer,
			uploadSize);

		stagingBuffer.map();
		memcpy(stagingBuffer.mapped, fontData, uploadSize);
		stagingBuffer.unmap();

		// Copy buffer data to font image
		vk::CommandBuffer copyCmd = device->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		// Prepare for transfer
		vks::tools::setImageLayout(
			copyCmd,
			fontImage,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			vk::PipelineStageFlagBits::eHost,
			vk::PipelineStageFlagBits::eTransfer);

		// Copy
		vk::BufferImageCopy bufferCopyRegion = {};
		bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		bufferCopyRegion.imageSubresource.layerCount = 1;
		bufferCopyRegion.imageExtent.width = texWidth;
		bufferCopyRegion.imageExtent.height = texHeight;
		bufferCopyRegion.imageExtent.depth = 1;

		copyCmd.copyBufferToImage(
			stagingBuffer.buffer,
			fontImage,
			vk::ImageLayout::eTransferDstOptimal,
			bufferCopyRegion
		);

		// Prepare for shader read
		vks::tools::setImageLayout(
			copyCmd,
			fontImage,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader);

		device->flushCommandBuffer(copyCmd, copyQueue, true);

		stagingBuffer.destroy();

		// Font texture Sampler
		vk::SamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
		sampler = device->logicalDevice.createSampler(samplerInfo);

		// Descriptor pool
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1)
		};
		vk::DescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		descriptorPool = device->logicalDevice.createDescriptorPool(descriptorPoolInfo);

		// Descriptor set layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, 0),
		};
		vk::DescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayout = device->logicalDevice.createDescriptorSetLayout(descriptorLayout);

		// Descriptor set
		vk::DescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		descriptorSet = device->logicalDevice.allocateDescriptorSets(allocInfo)[0];
		vk::DescriptorImageInfo fontDescriptor = vks::initializers::descriptorImageInfo(
			sampler,
			fontView,
			vk::ImageLayout::eShaderReadOnlyOptimal
		);
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eCombinedImageSampler, 0, &fontDescriptor)
		};
		device->logicalDevice.updateDescriptorSets(writeDescriptorSets, nullptr);

		// Pipeline cache
		vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};

		pipelineCache = device->logicalDevice.createPipelineCache(pipelineCacheCreateInfo);

		// Pipeline layout
		// Push constants for UI rendering parameters
		vk::PushConstantRange pushConstantRange = vks::initializers::pushConstantRange(vk::ShaderStageFlagBits::eVertex, sizeof(PushConstBlock), 0);
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayout = device->logicalDevice.createPipelineLayout(pipelineLayoutCreateInfo);

		// Setup graphics pipeline for UI rendering
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);

		// Enable blending
		vk::PipelineColorBlendAttachmentState blendAttachmentState{};
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{};

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Vertex bindings an attributes based on ImGui vertex definition
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(ImDrawVert), vk::VertexInputRate::eVertex),
		};
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, pos)),	// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, uv)),	// Location 1: UV
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR8G8B8A8Unorm, offsetof(ImDrawVert, col)),	// Location 0: Color
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		shaderStages[0] = example->loadShader(example->getAssetPath() + "shaders/ui.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = example->loadShader(example->getAssetPath() + "shaders/ui.frag.spv", vk::ShaderStageFlagBits::eFragment);

		pipeline = device->logicalDevice.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Starts a new imGui frame and sets up windows and ui elements
	void newFrame(VulkanExampleBase *example, bool updateFrameGraph)
	{
		ImGui::NewFrame();

		// Init imGui windows and elements

		//ImVec4 clear_color = ImColor(114, 144, 154);
		//static float f = 0.0f;
		ImGui::Text(example->title.c_str());
		ImGui::Text(device->properties.deviceName);

		// Update frame time display
		if (updateFrameGraph) {
			std::rotate(uiSettings.frameTimes.begin(), uiSettings.frameTimes.begin() + 1, uiSettings.frameTimes.end());
			float frameTime = 1000.0f / (example->frameTimer * 1000.0f);
			uiSettings.frameTimes.back() = frameTime;
			if (frameTime < uiSettings.frameTimeMin) {
				uiSettings.frameTimeMin = frameTime;
			}
			if (frameTime > uiSettings.frameTimeMax) {
				uiSettings.frameTimeMax = frameTime;
			}
		}

		ImGui::PlotLines("Frame Times", &uiSettings.frameTimes[0], 50, 0, "", uiSettings.frameTimeMin, uiSettings.frameTimeMax, ImVec2(0, 80));

		ImGui::Text("Camera");
		ImGui::InputFloat3("position", &example->camera.position.x, 2);
		ImGui::InputFloat3("rotation", &example->camera.rotation.x, 2);

		ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Example settings");
		ImGui::Checkbox("Render models", &uiSettings.displayModels);
		ImGui::Checkbox("Display logos", &uiSettings.displayLogos);
		ImGui::Checkbox("Display background", &uiSettings.displayBackground);
		ImGui::Checkbox("Animate light", &uiSettings.animateLight);
		ImGui::SliderFloat("Light speed", &uiSettings.lightSpeed, 0.1f, 1.0f);
		ImGui::End();

		ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
		ImGui::ShowTestWindow();

		// Render to generate draw buffers
		ImGui::Render();
	}

	// Update vertex and index buffer containing the imGui elements when required
	void updateBuffers()
	{
		ImDrawData* imDrawData = ImGui::GetDrawData();

		// Note: Alignment is done inside buffer creation
		vk::DeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
		vk::DeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);

		// Update buffers only if vertex or index count has been changed compared to current buffer size

		// Vertex buffer
		if (!(vertexBuffer.buffer) || (vertexCount != imDrawData->TotalVtxCount)) {
			vertexBuffer.unmap();
			vertexBuffer.destroy();
			device->createBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eHostVisible, &vertexBuffer, vertexBufferSize);
			vertexCount = imDrawData->TotalVtxCount;
			vertexBuffer.unmap();
			vertexBuffer.map();
		}

		// Index buffer
		//vk::DeviceSize indexSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);
		if (!(indexBuffer.buffer) || (indexCount < imDrawData->TotalIdxCount)) {
			indexBuffer.unmap();
			indexBuffer.destroy();
			device->createBuffer(vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eHostVisible, &indexBuffer, indexBufferSize);
			indexCount = imDrawData->TotalIdxCount;
			indexBuffer.map();
		}

		// Upload data
		ImDrawVert* vtxDst = (ImDrawVert*)vertexBuffer.mapped;
		ImDrawIdx* idxDst = (ImDrawIdx*)indexBuffer.mapped;

		for (int n = 0; n < imDrawData->CmdListsCount; n++) {
			const ImDrawList* cmd_list = imDrawData->CmdLists[n];
			memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
			memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
			vtxDst += cmd_list->VtxBuffer.Size;
			idxDst += cmd_list->IdxBuffer.Size;
		}

		// Flush to make writes visible to GPU
		vertexBuffer.flush();
		indexBuffer.flush();
	}

	// Draw current imGui frame into a command buffer
	void drawFrame(vk::CommandBuffer commandBuffer)
	{
		ImGuiIO& io = ImGui::GetIO();

		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

		// Bind vertex and index buffer
		std::vector<vk::DeviceSize> offsets = { 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffer.buffer, offsets);
		commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);

		vk::Viewport viewport = vks::initializers::viewport(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f);
		commandBuffer.setViewport(0, viewport);

		// UI scale and translate via push constants
		pushConstBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
		pushConstBlock.translate = glm::vec2(-1.0f);
		commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstBlock), &pushConstBlock);

		// Render commands
		ImDrawData* imDrawData = ImGui::GetDrawData();
		int32_t vertexOffset = 0;
		int32_t indexOffset = 0;
		for (int32_t i = 0; i < imDrawData->CmdListsCount; i++)
		{
			const ImDrawList* cmd_list = imDrawData->CmdLists[i];
			for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++)
			{
				const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[j];
				vk::Rect2D scissorRect;
				scissorRect.offset.x = std::max((int32_t)(pcmd->ClipRect.x), 0);
				scissorRect.offset.y = std::max((int32_t)(pcmd->ClipRect.y), 0);
				scissorRect.extent.width = (uint32_t)(pcmd->ClipRect.z - pcmd->ClipRect.x);
				scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - pcmd->ClipRect.y);
				commandBuffer.setScissor(0, scissorRect);
				commandBuffer.drawIndexed(pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
				indexOffset += pcmd->ElemCount;
			}
			vertexOffset += cmd_list->VtxBuffer.Size;
		}
	}

};

// ----------------------------------------------------------------------------
// VulkanExample
// ----------------------------------------------------------------------------

class VulkanExample : public VulkanExampleBase
{
public:
	ImGUI *imGui = nullptr;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct Models {
		vks::Model models;
		vks::Model logos;
		vks::Model background;
	} models;

	vks::Buffer uniformBufferVS;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 modelview;
		glm::vec4 lightPos;
	} uboVS;

	vk::PipelineLayout pipelineLayout;
	vk::Pipeline pipeline;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::DescriptorSet descriptorSet;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - ImGui";
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 1.4f, -4.8f));
		camera.setRotation(glm::vec3(4.5f, -380.0f, 0.0f));
		camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		device.destroyPipeline(pipeline);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		models.models.destroy();
		models.background.destroy();
		models.logos.destroy();

		uniformBufferVS.destroy();

		delete imGui;
	}
	
	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo;

		vk::ClearValue clearValues[2];
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.2f, 0.2f, 0.2f, 1.0f} };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		imGui->newFrame(this, (frameCounter == 0));

		imGui->updateBuffers();

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			// Render scene
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

			std::vector<vk::DeviceSize> offsets = { 0 };
			if (uiSettings.displayBackground) {
				drawCmdBuffers[i].bindVertexBuffers(0, models.background.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.background.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].drawIndexed(models.background.indexCount, 1, 0, 0, 0);
			}

			if (uiSettings.displayModels) {
				drawCmdBuffers[i].bindVertexBuffers(0, models.models.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.models.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].drawIndexed(models.models.indexCount, 1, 0, 0, 0);
			}

			if (uiSettings.displayLogos) {
				drawCmdBuffers[i].bindVertexBuffers(0, models.logos.vertices.buffer, offsets);
				drawCmdBuffers[i].bindIndexBuffer(models.logos.indices.buffer, 0, vk::IndexType::eUint32);
				drawCmdBuffers[i].drawIndexed(models.logos.indexCount, 1, 0, 0, 0);
			}

			// Render imGui
			imGui->drawFrame(drawCmdBuffers[i]);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void setupLayoutsAndDescriptors()
	{
		// descriptor pool
		std::vector<vk::DescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1)
		};
		vk::DescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
		descriptorPool = device.createDescriptorPool(descriptorPoolInfo);

		// Set layout
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex, 0),
		};
		vk::DescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		// Pipeline layout
		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);

		// Descriptor set
		vk::DescriptorSetAllocateInfo allocInfo =	vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		descriptorSet = device.allocateDescriptorSets(allocInfo)[0];
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet, vk::DescriptorType::eUniformBuffer, 0, &uniformBufferVS.descriptor),
		};
		device.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	void preparePipelines()
	{
		// Rendering
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eFront, vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState();

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		std::vector<vk::VertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), vk::VertexInputRate::eVertex),
		};
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),					// Location 0: Position		
			vks::initializers::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),	// Location 1: Normal		
			vks::initializers::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 6),	// Location 2: Color		
		};
		vk::PipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipeline = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBufferVS,
			sizeof(uboVS),
			&uboVS);

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Vertex shader		
		uboVS.projection = camera.matrices.perspective;
		uboVS.modelview = camera.matrices.view * glm::mat4();

		// Light source
		if (uiSettings.animateLight) {
			uiSettings.lightTimer += frameTimer * uiSettings.lightSpeed;
			uboVS.lightPos.x = sin(glm::radians(uiSettings.lightTimer * 360.0f)) * 15.0f;
			uboVS.lightPos.z = cos(glm::radians(uiSettings.lightTimer * 360.0f)) * 15.0f;
		};

		uniformBufferVS.map();
		memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
		uniformBufferVS.unmap();
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();
		buildCommandBuffers();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		queue.submit(submitInfo, vk::Fence(nullptr));
		VulkanExampleBase::submitFrame();
	}

	void loadAssets()
	{
		models.models.loadFromFile(getAssetPath() + "models/vulkanscenemodels.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		models.background.loadFromFile(getAssetPath() + "models/vulkanscenebackground.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		models.logos.loadFromFile(getAssetPath() + "models/vulkanscenelogos.dae", vertexLayout, 1.0f, vulkanDevice, queue);
	}

	void prepareImGui()
	{
		imGui = new ImGUI(this);
		imGui->init((float)width, (float)height);
		imGui->initResources(renderPass, queue);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		setupLayoutsAndDescriptors();
		preparePipelines();
		prepareImGui();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;

		// Update imGui
		ImGuiIO& io = ImGui::GetIO();

		io.DisplaySize = ImVec2((float)width, (float)height);
		io.DeltaTime = frameTimer;

		// todo: Android touch/gamepad, different platforms
#if defined(_WIN32)
		io.MousePos = ImVec2(mousePos.x, mousePos.y);
		io.MouseDown[0] = (((GetKeyState(VK_LBUTTON) & 0x100) != 0));
		io.MouseDown[1] = (((GetKeyState(VK_RBUTTON) & 0x100) != 0));
#else
#endif

		draw();

		if (uiSettings.animateLight)
			updateUniformBuffers();
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()
