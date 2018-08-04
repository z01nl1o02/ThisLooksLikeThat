#pragma once

#ifdef chelp_EXPORTS
#define CHELP_API __declspec(dllexport)
#else
#define CHELP_API __declspec(dllimport)
#endif
