// client\src\hooks\useNextSequence.ts
import { useEffect, useState } from "react";
import { useList } from "@pankod/refine-core";

interface UseNextSequenceOptions {
  resource: string;
  type: "Create" | "Edit"; // Explicitly set the union type
  initialValues?: { seq?: number };
}

const useNextSequence = ({ resource, type, initialValues }: UseNextSequenceOptions) => {
  const [currentSeq, setCurrentSeq] = useState<number | null>(null);

  // Fetch existing data from the specified resource
  const { data: resourceData, isLoading } = useList({
    resource,
    config: {
      pagination: { pageSize: 100000 },
      sort: [{ field: "seq", order: "desc" }],
    },
    ...(type === "Create" ? {} : { filters: [] }),
  });

  // Get the next sequence number (highest + 1)
  const getNextSequence = (existingSeqs: number[]): number => {
    if (existingSeqs.length === 0) return 1;
    const highestSeq = Math.max(...existingSeqs);
    return highestSeq + 1;
  };

  useEffect(() => {
    if (type === "Edit" && initialValues?.seq) {
      setCurrentSeq(initialValues.seq);
    }
  }, [type, initialValues]);

  useEffect(() => {
    if (type === "Create" && resourceData?.data) {
      const existingSeqs = resourceData.data.map((item: any) => item.seq);
      const nextSeq = getNextSequence(existingSeqs);
      setCurrentSeq(nextSeq);
    }
  }, [type, resourceData]);

  return { currentSeq, isLoading };
};

export default useNextSequence;
