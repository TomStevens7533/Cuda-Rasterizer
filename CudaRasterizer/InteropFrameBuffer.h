#pragma once
#include <memory>
struct FrameBufferSpecific {
	//specification for a Framebuffer
	uint32_t Width;
	uint32_t Height;
	uint32_t Samples = 1;

	bool SwapChainTarget = false;
};
class InteropGLFrameBuffer 
{
public:
	InteropGLFrameBuffer(const FrameBufferSpecific& spec);
	~InteropGLFrameBuffer();
	void Invalidate();

	void Bind() ;
	void UnBind() ;

	inline uint32_t GetColorAttachment() const  { return m_ColorAttached; }
	inline uint32_t GetDepthAttachment() const  { return m_DepthAttachment; }
	inline uint32_t GetRenderID() const { return m_RendererID; }

	inline FrameBufferSpecific& GetSpecification()  { return m_FrameBufferSpecifics; }
	void Resize(uint32_t width, uint32_t height) ;
private:
	uint32_t m_RendererID = 0;
	uint32_t m_ColorAttached = 0, m_DepthAttachment = 0;
	FrameBufferSpecific m_FrameBufferSpecifics;
};