#include "stdio.h"
#include "stdlib.h"
#include "patch2col.h"



void patch2col_cpu(const float* data_img, const int channels, const int height, const int width, float* col_data)
{
    const int ks = 3;
	int out_width = width - ks + 1;
    int col_width = ks * ks * channels;
    int ch_page_size = width * height;
	for (int y = 0; y < height - ks + 1; y++) {
		for (int x = 0; x < width - ks + 1; x++) {
			float* out = col_data + (y * out_width + x) * col_width;
			int offset = 0;
			for (int chidx = 0; chidx < channels; chidx++) {
				const float* in = data_img + chidx * ch_page_size;
				for (int dy = 0; dy < ks; dy++) {
					for (int dx = 0; dx < ks; dx++) {
						out[offset] = in[(y + dy) * width + (x + dx)];
						offset++;
					}
				}
			}
		}
	}
    return;
}

void patch2col(const int dev, const float* data_img, const int channels, const int height, const int width, float* col_data)
{
    if(dev == 0)
    {
        patch2col_cpu(data_img, channels, height, width, col_data);
    }
    return;
}



void patch2col_2_cpu(const float* data_img, const int channels, const int height, const int width, float* col_data)
{
	const int ks = 3;
	int width_out = width - 2;
	int height_out  = height - 2;
	int page_size_out = channels * ks * ks;
	for(int y = 0; y < height_out; y++)
	{
		for(int x = 0; x < width_out; x++)
		{
			float* current_out = col_data + (y * width_out + x) * channels * ks * ks;
			int pos_out = 0;
			for(int ch = 0; ch < channels; ch++)
			{
				for(int yy = y; yy < y + ks; yy++)
				{
					for(int xx = x; xx < x + ks; xx++)
					{
						current_out[pos_out] = data_img[ch * width * height + yy * width + xx];
						pos_out ++;
					}
				}
			}
		}
	}	
	return;
}

void patch2col_2(const int dev, const float* data_img, const int channels, const int height, const int width, float* col_data)
{
	if(dev == 0)
	{
		patch2col_2_cpu(data_img, channels, height, width, col_data);
	}
	return;
}
