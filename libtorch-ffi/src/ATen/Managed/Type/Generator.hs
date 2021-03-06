
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.Generator where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Unmanaged.Type.Generator
import ATen.Unmanaged.Type.IntArray
import ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.Storage
import ATen.Unmanaged.Type.Tensor
import ATen.Unmanaged.Type.TensorList
import ATen.Unmanaged.Type.TensorOptions
import ATen.Unmanaged.Type.Tuple
import ATen.Unmanaged.Type.StdString
import ATen.Unmanaged.Type.Dimname
import ATen.Unmanaged.Type.DimnameList

import qualified ATen.Unmanaged.Type.Generator as Unmanaged



newCUDAGenerator
  :: Word16
  -> IO (ForeignPtr Generator)
newCUDAGenerator _device_index = cast1 Unmanaged.newCUDAGenerator _device_index

newCPUGenerator
  :: Word64
  -> IO (ForeignPtr Generator)
newCPUGenerator _seed_in = cast1 Unmanaged.newCPUGenerator _seed_in





generator_set_current_seed
  :: ForeignPtr Generator
  -> Word64
  -> IO ()
generator_set_current_seed = cast2 Unmanaged.generator_set_current_seed

generator_current_seed
  :: ForeignPtr Generator
  -> IO (Word64)
generator_current_seed = cast1 Unmanaged.generator_current_seed

generator_seed
  :: ForeignPtr Generator
  -> IO (Word64)
generator_seed = cast1 Unmanaged.generator_seed

generator_clone
  :: ForeignPtr Generator
  -> IO (ForeignPtr Generator)
generator_clone _obj = cast1 Unmanaged.generator_clone _obj
