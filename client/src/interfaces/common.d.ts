//client\src\interfaces\common.d.ts
import { ReactNode } from 'react';

export interface CustomButtonProps {
  type?: "button" | "submit" | "reset",
  title: string,
  backgroundColor: string,
  color: string,
  fullWidth?: boolean,
  icon?: ReactNode,
  disabled?: boolean,
  handleClick?: () => void
}

export interface CustomIconButtonProps {
  icon: ReactNode,
  title: string,
  backgroundColor: string,
  color: string,
  handleClick: () => void,
  size?: "small" | "medium" | "large"
}

export interface ProfileProps {
  type: string,
  name: string,
  avatar: string,
  email: string,
  properties: Array | undefined,
  isAdmin: boolean;
}

export interface PropertyProps {
  _id: string,
  title: string,
  description: string,
  location: string,
  price: string,
  photo: string,
  creator: string
}

export interface FormProps {
  type: string,
  register: any,
  onFinish: (values: FieldValues) => Promise<void | CreateResponse<BaseRecord> | UpdateResponse<BaseRecord>>,
  formLoading: boolean,
  handleSubmit: FormEventHandler<HTMLFormElement> | undefined,
  handleImageChange: (file) => void,
  onFinishHandler: (data: FieldValues) => Promise<void>,
  propertyImage?: { name: string, url: string },
}

export interface FormPropsProcurement {
  type: string,
  register: any,
  onFinish: (values: FieldValues) => Promise<void | CreateResponse<BaseRecord> | UpdateResponse<BaseRecord>>,
  formLoading: boolean,
  handleSubmit: FormEventHandler<HTMLFormElement> | undefined,
  onFinishHandler: (data: FieldValues) => Promise<void>,
  existingParts: Parts[],
  initialValues?: Record<string, any>;
}

export interface FormPropsSale {
  type: string,
  register: any,
  onFinish: (values: FieldValues) => Promise<void | CreateResponse<BaseRecord> | UpdateResponse<BaseRecord>>,
  formLoading: boolean,
  handleSubmit: FormEventHandler<HTMLFormElement> | undefined,
  onFinishHandler: (data: FieldValues) => Promise<void>,
  initialValues?: Record<string, any>;
}

interface FormPropsDeployment {
  type: string;
  register: any;
  onFinish: (values: FieldValues) => Promise<void | CreateResponse<BaseRecord> | UpdateResponse<BaseRecord>>,
  formLoading: boolean;
  handleSubmit: FormEventHandler<HTMLFormElement> | undefined,
  onFinishHandler: (data: FieldValues) => Promise<void>,
  existingParts: Parts[],
  initialValues?: Record<string, any>;
}

export interface FormPropsExpense {
  type: string,
  register: any,
  onFinish: (values: FieldValues) => Promise<void | CreateResponse<BaseRecord> | UpdateResponse<BaseRecord>>,
  formLoading: boolean,
  handleSubmit: FormEventHandler<HTMLFormElement> | undefined,
  onFinishHandler: (data: FieldValues) => Promise<void>,
  initialValues?: Record<string, any>;
}