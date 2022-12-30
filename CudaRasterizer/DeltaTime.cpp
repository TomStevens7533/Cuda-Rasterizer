#include "DeltaTime.h"
std::shared_ptr<Time> Time::m_TimeSingleton;
std::shared_ptr<Time> Time::GetInstance()
{

	if (m_TimeSingleton == nullptr) {
		//create singleton
		m_TimeSingleton = std::make_shared<Time>();
	}
	return m_TimeSingleton;
}

Time::Time()
{
	m_StartTime = std::chrono::high_resolution_clock::now();

}

float Time::GetDeltaTime()
{
	return m_DeltaTime;
}

void Time::Update()
{

	auto currentTime = std::chrono::high_resolution_clock::now();
	m_DeltaTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - m_StartTime).count();
	m_StartTime = currentTime;
}


