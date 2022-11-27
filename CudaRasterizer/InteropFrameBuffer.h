#pragma once
#include <memory>
struct FrameBufferSpecific {
	//specification for a Framebuffer
	uint32_t Width;
	uint32_t Height;
	uint32_t Samples = 1;

	bool SwapChainTarget = false;
};
class OpenGLFrameBuffer 
{
public:
	OpenGLFrameBuffer(const FrameBufferSpecific& spec);
	virtual ~OpenGLFrameBuffer();
	void Invalidate();

	virtual void Bind() override;
	virtual void UnBind() override;

	virtual inline uint32_t GetColorAttachment() const override { return m_ColorAttached; }
	virtual inline uint32_t GetDepthAttachment() const override { return m_DepthAttachment; }


	virtual inline FrameBufferSpecific& GetSpecification() override { return m_FrameBufferSpecifics; }


	virtual void Resize(uint32_t width, uint32_t height) override;

private:
	uint32_t m_RendererID = 0;
	uint32_t m_ColorAttached = 0, m_DepthAttachment = 0;
	FrameBufferSpecific m_FrameBufferSpecifics;
};