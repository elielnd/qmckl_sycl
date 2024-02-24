    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue queue = ctx->q;

    queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(type_nucl_num), [=](sycl::id<1> idx) {

		});
	});
    

    sycl::buffer<float, 1> buffer_x(&x, range<1>(1));
    sycl::accessor acc_x(buffer_x, h, read_write);