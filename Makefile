CONFIG  :=  ./config/Makefile.config

include $(CONFIG)

BUILD_PATH    :=  build
SRC_PATH      :=  src
INC_PATH      :=  include

CXX_SRC       :=  $(wildcard $(SRC_PATH)/*.cpp)

APP_OBJS      :=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))

APP_MKS       :=  $(APP_OBJS:.o=.mk)

APP_DEPS      :=  $(CXX_SRC)
APP_DEPS      +=  $(wildcard $(SRC_PATH)/*.h)

CXXFLAGS      :=  -std=c++17 -pthread -fPIC 

INCS          :=  -I $(SRC_PATH) \
				  -I $(OPENCV_INSTALL_DIR) \
				  -I $(INC_PATH) \
				  -I $(RKNN_INSTALL_DIR)/include \
				  -I $(RGA_INCLUDE_DIR) \
				  -I $(ICEORYX_INCLUDE_DIR) \
				  -I $(SPDLOG_INCLUDE_DIR)

LIBS          :=  -L /usr/local/lib \
				  -L $(RKNN_INSTALL_DIR)/aarch64 -lrknnrt\
				  -lstdc++fs -lfmt -lpthread \
				  -L /usr/lib/aarch64-linux-gnu/librga.so -lrga \
				  -liceoryx_posh -liceoryx_hoofs -liceoryx_binding_c -lzip\
				  `pkg-config --libs opencv4`


ifeq ($(DEBUG),1)
CXXFLAGS      +=  -g -O0 -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG
else
CXXFLAGS      +=  -O3
endif

ifeq ($(SHOW_WARNING),1)
CXXFLAGS      +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CXXFLAGS      +=  -w
endif


ifeq (, $(shell which bear))
BEARCMD       :=
else
ifeq (bear 3.0.18, $(shell bear --version))
BEARCMD       := bear --output config/compile_commands.json --
else
BEARCMD       := bear -o config/compile_commands.json
endif
endif


all: 
	@mkdir -p bin
	@$(BEARCMD) $(MAKE) -j16 --no-print-directory $(APP)
	@echo finished building $@. Have fun!!

run:
	@$(MAKE) --no-print-directory update
	@./bin/$(APP)

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o bin/$@ $(LIBS) $(INCS)

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

clean:
	rm -rf $(APP)
	rm -rf build
	rm -rf config/compile_commands.json
	rm -rf bin

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Compile CXX
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp 
	@echo Compile CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)
$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp
	@echo Compile Dependence CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 

.PHONY: all update show clean 