/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-13 00:00:00 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-13 02:34:21
 */

#ifndef __TINYOCL_H__
#define __TINYOCL_H__

#include <memory>

class TinyOCL {
public:
    class TinyOCLImpl;
    static TinyOCL &GetInstance();
    ~TinyOCL() = default;
    TinyOCL(const TinyOCL &) = delete;
    TinyOCL &operator=(const TinyOCL &) = delete;

private:
    TinyOCL();
    std::unique_ptr<TinyOCLImpl> impl_;
};

#endif // __TINYOCL_H__
