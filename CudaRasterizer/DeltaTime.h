#pragma once
#include <chrono>
#include <memory>
class Time final
{
public:
	static std::shared_ptr<Time> GetInstance();
	Time();
	~Time() = default;

	float GetDeltaTime();
	void Update();
private:
	static std::shared_ptr<Time> m_TimeSingleton;
	std::chrono::steady_clock::time_point  m_StartTime;
	float m_DeltaTime = 0.f;
};