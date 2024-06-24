/*
 * @Author: Zhou Zijian 
 * @Date: 2024-06-24 23:09:39 
 * @Last Modified by: Zhou Zijian
 * @Last Modified time: 2024-06-24 23:24:48
 */

#include <gtest/gtest.h>
#include <TinyOCL.h>
#include <vector>

TEST(TinyOCLTest, TestExecutorCreateKernel1)
{
    auto kernel = TinyOCL::Executor::GetInstance().CreateKernel("cl/calc.cl", "add", {});
    EXPECT_NE(kernel, nullptr);
}

TEST(TinyOCLTest, TestExecutorCreateKernel2)
{
    auto kernel = TinyOCL::Executor::GetInstance().CreateKernel("cl/calc.cl1", "add", {});
    EXPECT_EQ(kernel, nullptr);
}

TEST(TinyOCLTest, TestExecutorCreateKernel3)
{
    auto kernel = TinyOCL::Executor::GetInstance().CreateKernel("cl/calc1.cl", "add", {});
    EXPECT_EQ(kernel, nullptr);
}

TEST(TinyOCLTest, TestExecutorCreateKernel4)
{
    auto kernel = TinyOCL::Executor::GetInstance().CreateKernel("cl/calc.cl", "add1", {});
    EXPECT_EQ(kernel, nullptr);
}

TEST(TinyOCLTest, TestExecutorCreateBuffer1)
{
    auto buffer = TinyOCL::Executor::GetInstance().CreateBuffer(10 * sizeof(float));
    EXPECT_NE(buffer, nullptr);
}

TEST(TinyOCLTest, TestBuffer)
{
    size_t size = 10 * sizeof(int);
    auto buffer = TinyOCL::Executor::GetInstance().CreateBuffer(size);
    EXPECT_NE(buffer, nullptr);

    int *data = buffer->GetHostPtr<int *>();
    EXPECT_NE(data, nullptr);

    cl_mem mem = buffer->GetClMem();
    EXPECT_NE(mem, nullptr);

    size_t size1 = buffer->GetSize();
    EXPECT_EQ(size, size1);

    std::vector<int> host_data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    bool ret = buffer->Memcpy(host_data1.data(), size, TinyOCL::MemcpyKind::HostToDevice);
    EXPECT_EQ(ret, true);

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], host_data1[i]);
    }

    std::vector<int> host_data2(10, 0);
    ret = buffer->Memcpy(host_data2.data(), size, TinyOCL::MemcpyKind::DeviceToHost);
    EXPECT_EQ(ret, true);

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], host_data2[i]);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}