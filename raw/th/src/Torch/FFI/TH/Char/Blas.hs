{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Blas where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_swap :  n x incx y incy -> void
foreign import ccall "THBlas.h THCharBlas_swap"
  c_swap_ :: CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_swap_ with unused argument (for CTHState) to unify backpack signatures.
c_swap = const c_swap_

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h THCharBlas_scal"
  c_scal_ :: CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_scal_ with unused argument (for CTHState) to unify backpack signatures.
c_scal = const c_scal_

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h THCharBlas_copy"
  c_copy_ :: CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy = const c_copy_

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h THCharBlas_axpy"
  c_axpy_ :: CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_axpy_ with unused argument (for CTHState) to unify backpack signatures.
c_axpy = const c_axpy_

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h THCharBlas_dot"
  c_dot_ :: CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO CChar

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot = const c_dot_

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THCharBlas_gemv"
  c_gemv_ :: CChar -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_gemv_ with unused argument (for CTHState) to unify backpack signatures.
c_gemv = const c_gemv_

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THCharBlas_ger"
  c_ger_ :: CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_ger_ with unused argument (for CTHState) to unify backpack signatures.
c_ger = const c_ger_

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THCharBlas_gemm"
  c_gemm_ :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_gemm_ with unused argument (for CTHState) to unify backpack signatures.
c_gemm = const c_gemm_

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THCharBlas_swap"
  p_swap_ :: FunPtr (CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_swap_ with unused argument (for CTHState) to unify backpack signatures.
p_swap = const p_swap_

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THCharBlas_scal"
  p_scal_ :: FunPtr (CLLong -> CChar -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_scal_ with unused argument (for CTHState) to unify backpack signatures.
p_scal = const p_scal_

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THCharBlas_copy"
  p_copy_ :: FunPtr (CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_copy_ with unused argument (for CTHState) to unify backpack signatures.
p_copy = const p_copy_

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THCharBlas_axpy"
  p_axpy_ :: FunPtr (CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_axpy_ with unused argument (for CTHState) to unify backpack signatures.
p_axpy = const p_axpy_

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THCharBlas_dot"
  p_dot_ :: FunPtr (CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO CChar)

-- | alias of p_dot_ with unused argument (for CTHState) to unify backpack signatures.
p_dot = const p_dot_

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THCharBlas_gemv"
  p_gemv_ :: FunPtr (CChar -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_gemv_ with unused argument (for CTHState) to unify backpack signatures.
p_gemv = const p_gemv_

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THCharBlas_ger"
  p_ger_ :: FunPtr (CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_ger_ with unused argument (for CTHState) to unify backpack signatures.
p_ger = const p_ger_

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THCharBlas_gemm"
  p_gemm_ :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ())

-- | alias of p_gemm_ with unused argument (for CTHState) to unify backpack signatures.
p_gemm = const p_gemm_