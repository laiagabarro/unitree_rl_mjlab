### CycloneDDS Installation
```
# CycloneDDS (C library)
git clone --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON ..
cmake --build . -j$(nproc)
sudo cmake --install .
cd ../..

# CycloneDDS-CXX (C++ binding, provides dds/dds.hpp)
git clone --branch 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds-cxx.git
cd cyclonedds-cxx && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr/local ..
cmake --build . -j$(nproc)
sudo cmake --install .
```

### CycloneDDS Local Interface (`cyclonedds.xml`)
```
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://cdds.io/config/cyclonedds.xsd">
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>lo</NetworkInterfaceAddress>
    </General>
  </Domain>
</CycloneDDS>
```
#### Add to `~/.bashrc`
```
# CycloneDDS
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```